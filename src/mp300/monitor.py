"""Reproduce MP300 monitoring checks and the final champion predictions.

Run from the repository root with ``python -m src.mp300.monitor``. Training
data is cleaned exactly once. The later and holdout files are already in the
model's units and are never passed through ``clean_train``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import sqlite3
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml

from .clean import FEATURE_COLUMNS, TARGET_COLUMN, clean_train


NUMERIC_INPUT_COLUMNS = (
    "age",
    "screen_hours",
    "apps_installed",
    "plan_gb_limit",
    "video_minutes",
    "wifi_hours",
)
MODEL_NAME = "mp300_gb_per_day"
EXPECTED_GROUPS = ("Urban", "Semi-urban", "Rural")


@dataclass(frozen=True)
class RegisteredModel:
    """A model loaded from the repository's registered MLflow artifact."""

    version: int
    run_id: str
    artifact_path: Path
    model: Any


def _numeric_values(values: pd.Series | Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain finite numeric values")
    return result


def quantile_bin_edges(reference: pd.Series, bins: int = 10) -> np.ndarray:
    """Return training quantile cuts with open-ended outer bins.

    Repeated quantile cut points are collapsed so every interval has positive
    width. The outer edges are infinite so later observations beyond the
    training minimum or maximum remain counted in the first or last bin.
    """
    ref = _numeric_values(reference, "reference")
    if bins < 1:
        raise ValueError("bins must be at least 1")
    quantiles = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
    interior = np.unique(quantiles[1:-1])
    return np.concatenate(([-np.inf], interior, [np.inf]))


def psi_details(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    smoothing: float = 0.5,
) -> tuple[float, int]:
    """Calculate PSI and the effective bin count using shared train cuts.

    Counts use NumPy's histogram convention: bins are left-closed/right-open,
    except the final bin includes its right edge. Additive 0.5-count smoothing
    is applied in every bin before normalizing, including empty bins.
    """
    ref = _numeric_values(reference, "reference")
    cur = _numeric_values(current, "current")
    if smoothing <= 0 or not np.isfinite(smoothing):
        raise ValueError("smoothing must be a positive finite number")

    edges = quantile_bin_edges(pd.Series(ref), bins=bins)
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    effective_bins = len(edges) - 1
    expected = (ref_counts + smoothing) / (ref_counts.sum() + smoothing * effective_bins)
    observed = (cur_counts + smoothing) / (cur_counts.sum() + smoothing * effective_bins)
    psi = float(np.sum((observed - expected) * np.log(observed / expected)))
    return psi, effective_bins


def psi_by_decile(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    smoothing: float = 0.5,
) -> float:
    """Return PSI for up to ``bins`` quantile intervals from training data."""
    return psi_details(reference, current, bins=bins, smoothing=smoothing)[0]


def date_split(
    later: pd.DataFrame, train_fraction: float = 0.70
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by date, record ID, and original row order, then split by rows.

    This matches ``src/mp300/train.py::chronological_split``. The original
    row order is an explicit final key so duplicate date/ID pairs remain
    deterministic.
    """
    required = {"date", "record_id"}
    missing = sorted(required - set(later.columns))
    if missing:
        raise ValueError(f"later.csv is missing split columns: {missing}")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    ordered = later.assign(_source_order=np.arange(len(later)))
    ordered = ordered.sort_values(
        ["date", "record_id", "_source_order"], kind="mergesort"
    ).drop(columns="_source_order")
    boundary = int(len(ordered) * train_fraction)
    if not 0 < boundary < len(ordered):
        raise ValueError("The chronological split needs rows on both sides")
    return ordered.iloc[:boundary].copy(), ordered.iloc[boundary:].copy()


def split_id_sha256(rows: pd.DataFrame) -> str:
    """Hash ordered record IDs as UTF-8, joined with LF and ending in LF."""
    if "record_id" not in rows:
        raise ValueError("Split rows are missing record_id")
    value = "".join(f"{record_id}\n" for record_id in rows["record_id"].astype(str))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_registered_model(
    registry_db: Path,
    *,
    version: int | None = None,
    alias: str | None = None,
    model_name: str = MODEL_NAME,
) -> RegisteredModel:
    """Resolve a version or alias in SQLite and load its saved model artifact.

    The repository already contains MLflow's registered artifact and SQLite
    database. Resolving their tables directly avoids adding MLflow as a new
    monitoring dependency.
    """
    if (version is None) == (alias is None):
        raise ValueError("Specify exactly one of version or alias")
    database = Path(registry_db).resolve()
    if not database.is_file():
        raise FileNotFoundError(f"MLflow SQLite database not found: {database}")

    uri = database.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        if alias is not None:
            row = connection.execute(
                "SELECT version FROM registered_model_aliases "
                "WHERE name = ? AND alias = ?",
                (model_name, alias),
            ).fetchone()
            if row is None:
                raise ValueError(f"Registered model {model_name!r} has no alias {alias!r}")
            resolved_version = int(row["version"])
        else:
            resolved_version = int(version)

        row = connection.execute(
            "SELECT version, status, storage_location, run_id FROM model_versions "
            "WHERE name = ? AND version = ?",
            (model_name, resolved_version),
        ).fetchone()

    if row is None:
        raise ValueError(
            f"Registered model {model_name!r} has no version {resolved_version}"
        )
    if row["status"] != "READY":
        raise ValueError(
            f"Registered model {model_name!r} version {resolved_version} "
            f"is not READY: {row['status']}"
        )

    artifact_path = Path(row["storage_location"])
    if not artifact_path.is_absolute():
        artifact_path = database.parent / artifact_path
    artifact_path = artifact_path.resolve()
    model_file = artifact_path / "model.pkl"
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Registered model artifact is missing: {model_file}"
        )
    return RegisteredModel(
        version=resolved_version,
        run_id=str(row["run_id"]),
        artifact_path=artifact_path,
        model=joblib.load(model_file),
    )


def predict(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Predict with exactly the feature order declared by the cleaning code."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction data is missing model features: {missing}")
    predictions = np.asarray(model.predict(frame.loc[:, FEATURE_COLUMNS]), dtype=float).reshape(-1)
    if len(predictions) != len(frame) or not np.isfinite(predictions).all():
        raise ValueError("Model returned an invalid prediction array")
    return predictions


def mean_absolute_error(actual: pd.Series, predicted: Sequence[float]) -> float:
    targets = _numeric_values(actual, TARGET_COLUMN)
    predictions = _numeric_values(predicted, "predictions")
    if len(targets) != len(predictions):
        raise ValueError("Target and prediction row counts do not match")
    return float(np.mean(np.abs(targets - predictions)))


def validate_prediction_output(
    output: pd.DataFrame,
    expected_record_ids: Sequence[str],
    expected_rows: int = 250,
) -> None:
    """Validate the exact holdout CSV schema, count, ID order, and values."""
    if list(output.columns) != ["record_id", "prediction"]:
        raise ValueError("Prediction output must have exactly record_id,prediction columns")
    if len(output) != expected_rows or len(expected_record_ids) != expected_rows:
        raise ValueError(f"Expected exactly {expected_rows} prediction rows")
    expected_ids = [str(value) for value in expected_record_ids]
    actual_ids = output["record_id"].astype(str).tolist()
    if actual_ids != expected_ids:
        raise ValueError("Prediction record IDs do not match the original holdout order")
    try:
        values = output["prediction"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Predictions must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("Predictions must all be finite numeric values")


def save_predictions(
    path: Path, holdout: pd.DataFrame, predictions: Sequence[float]
) -> None:
    if "record_id" not in holdout:
        raise ValueError("Holdout data is missing record_id")
    output = pd.DataFrame(
        {"record_id": holdout["record_id"].astype(str), "prediction": predictions}
    )
    validate_prediction_output(output, holdout["record_id"].astype(str).tolist())
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    saved = pd.read_csv(destination, dtype={"record_id": str})
    validate_prediction_output(saved, holdout["record_id"].astype(str).tolist())


def verify_prediction_parity(
    registered_model: Any,
    local_model_path: Path,
    features: pd.DataFrame,
    atol: float = 1e-12,
) -> float:
    """Check that the model.joblib API artifact matches the registry champion."""
    local_model = joblib.load(local_model_path)
    registered_predictions = predict(registered_model, features)
    local_predictions = predict(local_model, features)
    if not np.allclose(registered_predictions, local_predictions, rtol=0.0, atol=atol):
        difference = float(np.max(np.abs(registered_predictions - local_predictions)))
        raise ValueError(
            "models/model.joblib differs from the registered champion "
            f"(maximum prediction difference {difference})"
        )
    return float(np.max(np.abs(registered_predictions - local_predictions)))


def update_monitoring_findings(
    findings_path: Path,
    *,
    psi: dict[str, float],
    mae_by_group: dict[str, float],
    worst_served_group: str,
    mae_v1_on_check: float,
    mae_v2_on_check: float,
    decision: str,
) -> None:
    """Replace only the monitoring YAML section, preserving all prior bytes."""
    path = Path(findings_path)
    original = path.read_text(encoding="utf-8")
    previous = yaml.safe_load(original)
    marker = re.search(r"(?m)^monitoring:[ \t]*(?:#.*)?$", original)
    if marker is None:
        raise ValueError("reports/findings.yaml has no monitoring section")
    if set(psi) != set(NUMERIC_INPUT_COLUMNS):
        raise ValueError("Monitoring PSI must include the six numeric input features")
    if set(mae_by_group) != set(EXPECTED_GROUPS):
        raise ValueError("Monitoring MAEs must include Urban, Semi-urban, and Rural")
    if decision not in {"promote", "keep"}:
        raise ValueError("decision must be 'promote' or 'keep'")

    section = [
        "monitoring:",
        "  psi:  # cleaned train.csv vs later.csv, up to 10 train quantile bins",
    ]
    section.extend(f"    {name}: {psi[name]!r}" for name in NUMERIC_INPUT_COLUMNS)
    section.append("  most_drifted_feature: " + max(psi, key=psi.get))
    section.append("  mae_by_group:  # registered v1 on all later.csv rows")
    section.extend(f"    '{name}': {mae_by_group[name]!r}" for name in EXPECTED_GROUPS)
    section.extend(
        [
            f"  worst_served_group: {worst_served_group}",
            f"  mae_v1_on_check: {mae_v1_on_check!r}  # shared last 30%",
            f"  mae_v2_on_check: {mae_v2_on_check!r}  # shared last 30%",
            f"  decision: {decision}",
        ]
    )
    updated = original[: marker.start()] + "\n".join(section) + "\n"
    parsed = yaml.safe_load(updated)
    if parsed.get("data_issues") != previous.get("data_issues"):
        raise AssertionError("Writing monitoring changed the data_issues section")
    if parsed.get("tracking") != previous.get("tracking"):
        raise AssertionError("Writing monitoring changed the tracking section")
    path.write_text(updated, encoding="utf-8", newline="\n")


def _print_split(training_rows: pd.DataFrame, check_rows: pd.DataFrame) -> None:
    print(f"later.csv rows: {len(training_rows) + len(check_rows)}")
    print(f"v2 retraining portion: {len(training_rows)} rows")
    print(f"common comparison set: {len(check_rows)} rows")
    print(
        "training boundary:",
        training_rows.iloc[-1]["date"],
        training_rows.iloc[-1]["record_id"],
    )
    print(
        "comparison boundary:",
        check_rows.iloc[0]["date"],
        check_rows.iloc[0]["record_id"],
    )
    print("retraining IDs SHA-256:", split_id_sha256(training_rows))
    print("comparison IDs SHA-256:", split_id_sha256(check_rows))


def run_analysis(
    *,
    data_dir: Path = Path("data/raw"),
    registry_db: Path = Path("mlflow.db"),
    model_file: Path = Path("models/model.joblib"),
    findings_path: Path = Path("reports/findings.yaml"),
    prediction_output: Path = Path("predictions/holdout_predictions.csv"),
) -> dict[str, Any]:
    """Run the real-data checks and write findings and holdout predictions."""
    raw_train = pd.read_csv(data_dir / "train.csv")
    cleaned_train = clean_train(raw_train)  # one call; later and holdout stay untouched
    later = pd.read_csv(data_dir / "later.csv")
    missing = sorted(
        (set(NUMERIC_INPUT_COLUMNS) | {TARGET_COLUMN, "circle", "date", "record_id"})
        - set(later.columns)
    )
    if missing:
        raise ValueError(f"later.csv is missing monitoring columns: {missing}")

    psi: dict[str, float] = {}
    effective_bins: dict[str, int] = {}
    for feature in NUMERIC_INPUT_COLUMNS:
        psi[feature], effective_bins[feature] = psi_details(
            cleaned_train[feature], later[feature], bins=10, smoothing=0.5
        )

    training_rows, check_rows = date_split(later, train_fraction=0.70)
    v1 = load_registered_model(registry_db, version=1)
    v2 = load_registered_model(registry_db, version=2)
    champion = load_registered_model(registry_db, alias="champion")

    prior_findings = yaml.safe_load(Path(findings_path).read_text(encoding="utf-8"))
    tracking = prior_findings.get("tracking") or {}
    if tracking.get("registered_model_name") != MODEL_NAME:
        raise ValueError("findings.yaml tracking model name does not match the registry")
    if tracking.get("best_run_id") != v1.run_id:
        raise ValueError("Registered v1 run ID does not match findings.yaml tracking")

    group_predictions = predict(v1.model, later)
    group_mae: dict[str, float] = {}
    for group in EXPECTED_GROUPS:
        mask = later["circle"].eq(group).to_numpy()
        if not mask.any():
            raise ValueError(f"later.csv has no rows for circle {group!r}")
        group_mae[group] = mean_absolute_error(
            later.loc[mask, TARGET_COLUMN], group_predictions[mask]
        )
    if set(later["circle"].dropna().astype(str).unique()) != set(EXPECTED_GROUPS):
        raise ValueError("later.csv circle values differ from the expected three groups")
    worst_group = max(group_mae, key=group_mae.get)

    v1_check = mean_absolute_error(
        check_rows[TARGET_COLUMN], predict(v1.model, check_rows)
    )
    v2_check = mean_absolute_error(
        check_rows[TARGET_COLUMN], predict(v2.model, check_rows)
    )
    promote_v2 = v2_check <= 0.88 * v1_check
    expected_champion = 2 if promote_v2 else 1
    if champion.version != expected_champion:
        raise ValueError(
            "Registry champion alias does not match the computed 12% promotion rule: "
            f"expected version {expected_champion}, found {champion.version}"
        )

    holdout = pd.read_csv(data_dir / "holdout_features.csv", dtype={"record_id": str})
    if len(holdout) != 250:
        raise ValueError(f"Expected 250 holdout rows, found {len(holdout)}")
    max_delta = verify_prediction_parity(champion.model, model_file, holdout)
    holdout_predictions = predict(champion.model, holdout)

    # Write only after all integrity and promotion checks have passed.
    save_predictions(prediction_output, holdout, holdout_predictions)
    decision = "promote" if promote_v2 else "keep"
    update_monitoring_findings(
        findings_path,
        psi=psi,
        mae_by_group=group_mae,
        worst_served_group=worst_group,
        mae_v1_on_check=v1_check,
        mae_v2_on_check=v2_check,
        decision=decision,
    )

    print("PSI: cleaned train.csv vs later.csv; 10 reference quantiles requested")
    print("      repeated cuts collapsed; outer edges are -infinity/+infinity")
    print("      smoothing is +0.5 count per effective bin before normalization")
    for feature in NUMERIC_INPUT_COLUMNS:
        print(f"  {feature}: {psi[feature]:.12f} ({effective_bins[feature]} effective bins)")
    print("Most drifted feature:", max(psi, key=psi.get))
    _print_split(training_rows, check_rows)
    print("Registered v1 version/run:", v1.version, v1.run_id)
    print("Registered v2 version/run:", v2.version, v2.run_id)
    print("v1 MAE on all later.csv rows by circle:")
    for group in EXPECTED_GROUPS:
        print(f"  {group}: {group_mae[group]:.12f} GB")
    print("Worst-served group:", worst_group)
    print(f"v1 MAE on common final 90 rows: {v1_check:.16f} GB")
    print(f"v2 MAE on common final 90 rows: {v2_check:.16f} GB")
    print("Promotion decision:", decision)
    print("Registered champion version:", champion.version)
    print(f"Registered champion vs model.joblib max prediction delta: {max_delta:.3g}")
    print(f"Saved {len(holdout_predictions)} holdout predictions to {prediction_output}")

    return {
        "psi": psi,
        "effective_bins": effective_bins,
        "mae_by_group": group_mae,
        "worst_served_group": worst_group,
        "mae_v1_on_check": v1_check,
        "mae_v2_on_check": v2_check,
        "decision": decision,
        "champion_version": champion.version,
        "training_ids_sha256": split_id_sha256(training_rows),
        "comparison_ids_sha256": split_id_sha256(check_rows),
        "holdout_rows": len(holdout_predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--registry-db", type=Path, default=Path("mlflow.db"))
    parser.add_argument("--model-file", type=Path, default=Path("models/model.joblib"))
    parser.add_argument("--findings", type=Path, default=Path("reports/findings.yaml"))
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path("predictions/holdout_predictions.csv"),
    )
    args = parser.parse_args()
    run_analysis(
        data_dir=args.data_dir,
        registry_db=args.registry_db,
        model_file=args.model_file,
        findings_path=args.findings,
        prediction_output=args.prediction_output,
    )


if __name__ == "__main__":
    main()
