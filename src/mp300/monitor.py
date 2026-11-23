"""Reproducible MP300 drift, model-check, and holdout calculations.

Run from the repository root.  The training cleaner is applied once to the
original training CSV; later and holdout rows are already in model units.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mp300.clean import FEATURE_COLUMNS, TARGET_COLUMN, clean_train


NUMERIC_INPUT_COLUMNS = [
    "age",
    "screen_hours",
    "apps_installed",
    "plan_gb_limit",
    "video_minutes",
    "wifi_hours",
]
DEFAULT_CHAMPION_MODEL = "models/model.joblib"


def _numeric_values(values: pd.Series, label: str) -> np.ndarray:
    result = values.to_numpy(dtype=float)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain finite numeric values")
    return result


def psi_by_decile(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI using reference-population quantile bins.

    The same decile edges from cleaned train are applied to later.  A 0.5-row
    additive correction keeps empty bins finite in the logarithm.
    """
    ref = _numeric_values(reference, "reference")
    cur = _numeric_values(current, "current")
    quantiles = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
    if np.unique(quantiles).size != bins + 1:
        raise ValueError(
            f"Cannot make {bins} distinct reference deciles for this feature"
        )

    edges = np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))
    ref_binned = pd.cut(ref, bins=edges, include_lowest=True)
    cur_binned = pd.cut(cur, bins=edges, include_lowest=True)
    categories = ref_binned.categories
    ref_counts = (
        pd.Series(ref_binned).value_counts(sort=False).reindex(categories, fill_value=0)
    ).to_numpy(dtype=float)
    cur_counts = (
        pd.Series(cur_binned).value_counts(sort=False).reindex(categories, fill_value=0)
    ).to_numpy(dtype=float)

    smoothing = 0.5
    expected = (ref_counts + smoothing) / (ref_counts.sum() + smoothing * bins)
    observed = (cur_counts + smoothing) / (cur_counts.sum() + smoothing * bins)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def date_split(later: pd.DataFrame, train_fraction: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a stable chronological 70/30 split, with ID as the tie-breaker."""
    required = {"date", "record_id"}
    missing = sorted(required - set(later.columns))
    if missing:
        raise ValueError(f"later.csv is missing split columns: {missing}")

    ordered = later.assign(_source_order=np.arange(len(later)))
    ordered = ordered.sort_values(
        ["date", "record_id", "_source_order"], kind="mergesort"
    ).drop(columns="_source_order")
    ordered = ordered.reset_index(drop=True)
    cut = int(len(ordered) * train_fraction)
    if cut == 0 or cut == len(ordered):
        raise ValueError("The 70/30 split needs rows on both sides")
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def _load_model(reference: str) -> Any:
    path = Path(reference)
    if path.is_file():
        import joblib

        return joblib.load(path)

    if reference.startswith(("models:/", "runs:/")):
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        return mlflow.sklearn.load_model(reference)

    raise FileNotFoundError(
        f"Model '{reference}' is neither a local file nor an MLflow model URI"
    )


def _predict(model: Any, frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction data is missing model features: {missing}")
    predictions = np.asarray(model.predict(frame[FEATURE_COLUMNS]), dtype=float).reshape(-1)
    if len(predictions) != len(frame) or not np.isfinite(predictions).all():
        raise ValueError("Model returned an invalid prediction array")
    return predictions


def _mae(actual: pd.Series, predicted: np.ndarray) -> float:
    target = _numeric_values(actual, TARGET_COLUMN)
    return float(np.mean(np.abs(target - predicted)))


def _print_split(first: pd.DataFrame, last: pd.DataFrame) -> None:
    print(f"later.csv rows: {len(first) + len(last)}")
    print(f"70% v2 training rows: {len(first)}")
    print(f"30% shared comparison rows: {len(last)}")
    print(
        "70% ends at:",
        first.iloc[-1]["date"],
        first.iloc[-1]["record_id"],
    )
    print(
        "30% starts at:",
        last.iloc[0]["date"],
        last.iloc[0]["record_id"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-model", help="joblib file or MLflow URI for Hardik's v1")
    parser.add_argument("--v2-model", help="joblib file or MLflow URI for Hardik's v2")
    parser.add_argument(
        "--champion-model",
        help=f"joblib file or MLflow URI; defaults to {DEFAULT_CHAMPION_MODEL} if present",
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path("predictions/holdout_predictions.csv"),
    )
    args = parser.parse_args()

    data_dir = Path("data/raw")
    raw_train = pd.read_csv(data_dir / "train.csv")
    cleaned_train = clean_train(raw_train)  # exactly once, and only on train.csv
    later = pd.read_csv(data_dir / "later.csv")

    psi = {
        feature: psi_by_decile(cleaned_train[feature], later[feature])
        for feature in NUMERIC_INPUT_COLUMNS
    }
    print("PSI (cleaned train vs later, 10 training deciles):")
    for feature, value in psi.items():
        print(f"  {feature}: {value:.10f}")
    most_drifted = max(psi, key=psi.get)
    print("Most drifted feature:", most_drifted)

    first_70, last_30 = date_split(later)
    _print_split(first_70, last_30)

    if args.v1_model:
        v1 = _load_model(args.v1_model)
        print("v1 MAE by circle on all later.csv rows:")
        for circle, rows in later.groupby("circle", sort=True):
            score = _mae(rows[TARGET_COLUMN], _predict(v1, rows))
            print(f"  {circle}: {score:.6f}")
        mae_v1 = _mae(last_30[TARGET_COLUMN], _predict(v1, last_30))
        print(f"v1 MAE on shared last 30%: {mae_v1:.6f}")

        if args.v2_model:
            v2 = _load_model(args.v2_model)
            mae_v2 = _mae(last_30[TARGET_COLUMN], _predict(v2, last_30))
            promote = mae_v2 <= 0.88 * mae_v1
            decision = "promote v2" if promote else "keep v1"
            print(f"v2 MAE on same last 30%: {mae_v2:.6f}")
            print(f"Promotion rule (v2 <= 0.88 * v1): {decision}")
    else:
        print("Model MAEs skipped: Hardik's v1/v2 artifacts are not available yet.")

    champion_reference = args.champion_model
    if champion_reference is None and Path(DEFAULT_CHAMPION_MODEL).is_file():
        champion_reference = DEFAULT_CHAMPION_MODEL
    if champion_reference is None:
        print("Holdout predictions skipped: the final champion artifact is unavailable.")
        return

    holdout = pd.read_csv(data_dir / "holdout_features.csv")
    if len(holdout) != 250:
        raise ValueError(f"Expected 250 holdout rows, found {len(holdout)}")
    champion = _load_model(champion_reference)
    predictions = _predict(champion, holdout)
    output = pd.DataFrame(
        {"record_id": holdout["record_id"], "prediction": predictions}
    )
    if list(output.columns) != ["record_id", "prediction"]:
        raise AssertionError("Prediction columns do not match the required order")
    if not output["record_id"].reset_index(drop=True).equals(
        holdout["record_id"].reset_index(drop=True)
    ):
        raise AssertionError("Holdout record order changed")
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.prediction_output, index=False)
    saved = pd.read_csv(args.prediction_output)
    if len(saved) != 250 or list(saved.columns) != ["record_id", "prediction"]:
        raise AssertionError("Saved prediction file failed its shape/header check")
    if not saved["record_id"].equals(holdout["record_id"]):
        raise AssertionError("Saved predictions do not preserve holdout ID order")
    if not np.isfinite(saved["prediction"].to_numpy(dtype=float)).all():
        raise AssertionError("Saved predictions contain a missing or non-finite value")
    print(f"Saved {len(saved)} holdout predictions to {args.prediction_output}")


if __name__ == "__main__":
    main()
