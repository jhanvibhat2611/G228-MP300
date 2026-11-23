"""Train reproducible MP300 v1/v2 models when the team artifacts are absent.

Run from the repository root.  v1 uses a chronological 80/20 validation split
of cleaned train.csv.  v2 reuses the selected settings and adds the first 70%
of later.csv; its last 30% is held out for the promotion comparison.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mp300.clean import FEATURE_COLUMNS, TARGET_COLUMN, clean_train
from mp300.monitor import date_split


NUMERIC_COLUMNS = [
    "age",
    "screen_hours",
    "apps_installed",
    "plan_gb_limit",
    "video_minutes",
    "wifi_hours",
]
CATEGORICAL_COLUMNS = ["phone_tier", "circle"]
MODEL_NAME = "mp300_gb_per_day"
V1_EXPERIMENT = "MP300-v1-training"
V2_EXPERIMENT = "MP300-v2-retraining"


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_settings() -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "random_forest",
            {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "max_features": 1.0,
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        (
            "extra_trees",
            {
                "n_estimators": 400,
                "min_samples_leaf": 2,
                "max_features": 0.8,
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        (
            "hist_gradient_boosting",
            {
                "max_iter": 300,
                "learning_rate": 0.05,
                "max_leaf_nodes": 15,
                "l2_regularization": 1.0,
                "early_stopping": False,
                "random_state": 42,
            },
        ),
    ]


def make_pipeline(kind: str, params: dict[str, Any]) -> Pipeline:
    estimators = {
        "random_forest": RandomForestRegressor,
        "extra_trees": ExtraTreesRegressor,
        "hist_gradient_boosting": HistGradientBoostingRegressor,
    }
    if kind not in estimators:
        raise ValueError(f"Unknown model setting: {kind}")
    return Pipeline(
        [
            ("preprocess", _preprocessor()),
            ("regressor", estimators[kind](**params)),
        ]
    )


def _log_common(data_hash: str, kind: str, params: dict[str, Any]) -> None:
    mlflow.log_param("model_setting", kind)
    mlflow.log_param("data_sha256", data_hash)
    mlflow.log_params({f"model_{key}": value for key, value in params.items()})


def _register_model(
    pipeline: Pipeline,
    run_name: str,
    kind: str,
    data_hash: str,
    model_params: dict[str, Any],
    run_params: dict[str, Any],
    metrics: dict[str, float],
) -> tuple[str, int]:
    with mlflow.start_run(run_name=run_name) as run:
        _log_common(data_hash, kind, model_params)
        mlflow.log_params(run_params)
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle",
            await_registration_for=300,
        )
        client = MlflowClient()
        version = getattr(info, "registered_model_version", None)
        if version is None:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            version = max(int(item.version) for item in versions)
        return run.info.run_id, int(version)


def _ensure_fresh_registry(client: MlflowClient) -> None:
    existing = [model.name for model in client.search_registered_models()]
    if MODEL_NAME in existing:
        raise RuntimeError(
            f"Registry model {MODEL_NAME!r} already exists. Do not rerun this script "
            "over an existing registry; use the existing v1/v2 artifacts instead."
        )


def main() -> None:
    root = Path.cwd()
    raw_path = root / "data/raw/train.csv"
    later_path = root / "data/raw/later.csv"
    raw_train = pd.read_csv(raw_path)
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    # Apply the team's corrections once to the original training file only.
    train = clean_train(raw_train)
    train = train.sort_values(["date", "record_id"], kind="mergesort").reset_index(drop=True)
    cut = int(len(train) * 0.80)
    fit_rows, validation_rows = train.iloc[:cut], train.iloc[cut:]
    x_fit, y_fit = fit_rows[FEATURE_COLUMNS], fit_rows[TARGET_COLUMN]
    x_validation, y_validation = validation_rows[FEATURE_COLUMNS], validation_rows[TARGET_COLUMN]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    _ensure_fresh_registry(client)
    v1_experiment = mlflow.set_experiment(V1_EXPERIMENT)

    scores: list[tuple[float, str, dict[str, Any]]] = []
    for kind, params in candidate_settings():
        pipeline = make_pipeline(kind, params)
        pipeline.fit(x_fit, y_fit)
        predicted = pipeline.predict(x_validation)
        val_mae = float(mean_absolute_error(y_validation, predicted))
        with mlflow.start_run(experiment_id=v1_experiment.experiment_id, run_name=f"v1-{kind}"):
            _log_common(raw_hash, kind, params)
            mlflow.log_params(
                {
                    "validation_strategy": "chronological_80_20",
                    "fit_rows": len(fit_rows),
                    "validation_rows": len(validation_rows),
                }
            )
            mlflow.log_metric("mae_validation", val_mae)
            mlflow.set_tag("phase", "v1_candidate")
        print(f"v1 candidate {kind}: validation MAE = {val_mae:.6f} GB")
        scores.append((val_mae, kind, params))

    best_v1_mae, best_kind, best_params = min(scores, key=lambda item: item[0])
    v1 = make_pipeline(best_kind, best_params)
    v1.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    v1_run_id, v1_version = _register_model(
        v1,
        "v1-registered-model",
        best_kind,
        raw_hash,
        best_params,
        {"validation_strategy": "chronological_80_20", "fit_rows": len(train)},
        {"mae_validation": best_v1_mae},
    )
    if v1_version != 1:
        raise RuntimeError(f"Expected v1 to register as version 1; got version {v1_version}")
    client.set_registered_model_alias(MODEL_NAME, "champion", str(v1_version))

    later = pd.read_csv(later_path)
    first_70, last_30 = date_split(later)
    v1_check_mae = float(
        mean_absolute_error(last_30[TARGET_COLUMN], v1.predict(last_30[FEATURE_COLUMNS]))
    )
    combined = pd.concat(
        [
            train[FEATURE_COLUMNS + [TARGET_COLUMN]],
            first_70[FEATURE_COLUMNS + [TARGET_COLUMN]],
        ],
        ignore_index=True,
    )
    v2 = make_pipeline(best_kind, best_params)
    v2.fit(combined[FEATURE_COLUMNS], combined[TARGET_COLUMN])
    v2_check_mae = float(
        mean_absolute_error(last_30[TARGET_COLUMN], v2.predict(last_30[FEATURE_COLUMNS]))
    )
    v2_experiment = mlflow.set_experiment(V2_EXPERIMENT)
    with mlflow.start_run(experiment_id=v2_experiment.experiment_id, run_name="v2-retrained"):
        _log_common(raw_hash, best_kind, best_params)
        mlflow.log_params(
            {
                "train_rows_original": len(train),
                "later_rows_added": len(first_70),
                "check_rows": len(last_30),
                "later_split": "date_then_record_id_70_30",
            }
        )
        mlflow.log_metrics(
            {
                "mae_v1_on_check": v1_check_mae,
                "mae_v2_on_check": v2_check_mae,
            }
        )
        mlflow.set_tag("phase", "v2_retraining")
        info = mlflow.sklearn.log_model(
            v2,
            name="model",
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle",
            await_registration_for=300,
        )
        v2_run_id = mlflow.active_run().info.run_id

    v2_version = getattr(info, "registered_model_version", None)
    if v2_version is None:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        v2_version = max(int(item.version) for item in versions)
    v2_version = int(v2_version)
    if v2_version != 2:
        raise RuntimeError(f"Expected v2 to register as version 2; got version {v2_version}")

    promote = v2_check_mae <= 0.88 * v1_check_mae
    champion_version = v2_version if promote else v1_version
    client.set_registered_model_alias(MODEL_NAME, "champion", str(champion_version))

    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(v1, model_dir / "v1.joblib")
    joblib.dump(v2, model_dir / "v2.joblib")
    champion = v2 if promote else v1
    joblib.dump(champion, model_dir / "model.joblib")

    print(f"Selected v1 setting: {best_kind}")
    print(f"v1 validation MAE: {best_v1_mae:.6f} GB")
    print(f"v1 MAE on shared last 30%: {v1_check_mae:.6f} GB")
    print(f"v2 MAE on same last 30%: {v2_check_mae:.6f} GB")
    print("Promotion rule (v2 <= 0.88 * v1):", "promote v2" if promote else "keep v1")
    print(f"Registered model: {MODEL_NAME}; v1={v1_version}; v2={v2_version}")
    print(f"Champion version: {champion_version}")
    print(f"v1 registered run id: {v1_run_id}")
    print(f"v2 run id: {v2_run_id}")
    print(f"Original train.csv sha256: {raw_hash}")


if __name__ == "__main__":
    main()
