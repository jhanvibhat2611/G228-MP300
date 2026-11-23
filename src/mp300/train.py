"""Train MP300 v1 and v2 from the repository root.

Run with ``python -m src.mp300.train`` after copying the original train.csv
and later.csv into data/raw. Use a fresh registry for the initial v1/v2 pair.
"""

import hashlib
from io import BytesIO
from pathlib import Path
import sqlite3
from urllib.parse import unquote

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .clean import FEATURE_COLUMNS, TARGET_COLUMN, clean_train


EXPERIMENT_NAME = "G228-MP300-training"
MODEL_NAME = "mp300_gb_per_day"
MODEL_DIR = Path("models")
MLFLOW_DB = Path("mlflow.db")
FINDINGS_PATH = Path("reports/findings.yaml")
CATEGORICAL_COLUMNS = ["phone_tier", "circle"]
NUMERIC_COLUMNS = [name for name in FEATURE_COLUMNS if name not in CATEGORICAL_COLUMNS]
PROMOTION_RATIO = 0.88


def chronological_split(rows: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Match Junaid's date, record ID, then original row order tie-breakers.
    ordered = rows.assign(_source_order=range(len(rows))).sort_values(
        ["date", "record_id", "_source_order"]
    ).drop(columns="_source_order")
    boundary = int(len(ordered) * fraction)
    if not 0 < boundary < len(ordered):
        raise ValueError("The chronological split needs rows on both sides.")
    return ordered.iloc[:boundary], ordered.iloc[boundary:]


def make_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_COLUMNS),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
    ])
    return Pipeline([("preprocess", preprocessor), ("regressor", clone(estimator))])


def score(model: Pipeline, rows: pd.DataFrame) -> float:
    return float(mean_absolute_error(rows[TARGET_COLUMN], model.predict(rows[FEATURE_COLUMNS])))


def log_training_parameters(data_sha256: str, model: Pipeline, **params) -> None:
    estimator = model.named_steps["regressor"]
    mlflow.log_params({
        "data_sha256": data_sha256,
        "estimator": type(estimator).__name__,
        **params,
        **{f"model_{name}": value for name, value in estimator.get_params().items()},
    })


def register_model(model: Pipeline, expected_version: int) -> str:
    info = mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
    version = mlflow.register_model(info.model_uri, MODEL_NAME).version
    if int(version) != expected_version:
        raise RuntimeError(f"Expected model version {expected_version}; MLflow returned {version}.")
    return version


def update_tracking_findings(tracking: dict) -> None:
    content = FINDINGS_PATH.read_text(encoding="utf-8")
    before, _, tracking_and_rest = content.partition("\ntracking:\n")
    _, following_heading, rest = tracking_and_rest.partition("\nmonitoring:\n")
    if not following_heading:
        raise ValueError("Expected tracking and monitoring sections in reports/findings.yaml.")
    updated_tracking = yaml.safe_dump({"tracking": tracking}, sort_keys=False)
    FINDINGS_PATH.write_text(f"{before}\n{updated_tracking}{following_heading}{rest}", encoding="utf-8")


def use_relative_artifact_paths() -> None:
    """Keep MLflow model artifacts loadable from another clone of this repo."""
    root = Path.cwd()
    prefixes = {f"{root.as_posix()}/", f"{root.as_uri()}/", f"{unquote(root.as_uri())}/"}
    path_columns = (
        ("experiments", "artifact_location"), ("runs", "artifact_uri"),
        ("logged_models", "artifact_location"), ("model_versions", "storage_location"),
    )
    with sqlite3.connect(MLFLOW_DB) as db:
        for table, column in path_columns:
            for prefix in prefixes:
                db.execute(
                    f"UPDATE {table} SET {column} = substr({column}, ?) "
                    f"WHERE substr({column}, 1, ?) = ?",
                    (len(prefix) + 1, len(prefix), prefix),
                )
        db.commit()
        db.execute("VACUUM")

    for metadata in (MODEL_DIR / "mlflow").glob("models/*/artifacts/MLmodel"):
        content = metadata.read_text(encoding="utf-8")
        for prefix in prefixes:
            content = content.replace(f"artifact_path: {prefix}", "artifact_path: ")
        metadata.write_text(content, encoding="utf-8")


def main() -> None:
    train_path = Path("data/raw/train.csv")
    later_path = Path("data/raw/later.csv")
    for path in (train_path, later_path):
        if not path.is_file():
            raise FileNotFoundError(f"Copy the original {path.name} from the group ZIP to {path}.")

    original_train = train_path.read_bytes()
    data_sha256 = hashlib.sha256(original_train).hexdigest()
    train = clean_train(pd.read_csv(BytesIO(original_train)))
    fit_rows, validation = chronological_split(train, 0.80)
    later_fit, later_check = chronological_split(pd.read_csv(later_path), 0.70)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    if client.search_registered_models(f"name = '{MODEL_NAME}'"):
        raise RuntimeError(f"{MODEL_NAME} already exists in mlflow.db. Use a fresh registry to train v1/v2.")
    MODEL_DIR.mkdir(exist_ok=True)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(MODEL_DIR / "mlflow").as_posix())
    mlflow.set_experiment(EXPERIMENT_NAME)

    settings = [
        ("ridge", Ridge(alpha=1.0)),
        ("random_forest", RandomForestRegressor(
            n_estimators=240, min_samples_leaf=2, random_state=42, n_jobs=1,
        )),
        ("hist_gradient_boosting", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=False, random_state=42,
        )),
    ]
    candidates = []
    for name, estimator in settings:
        model = make_pipeline(estimator)
        with mlflow.start_run(run_name=f"v1-validation-{name}"):
            log_training_parameters(data_sha256, model, fit_rows=len(fit_rows), validation_rows=len(validation))
            model.fit(fit_rows[FEATURE_COLUMNS], fit_rows[TARGET_COLUMN])
            mae = score(model, validation)
            mlflow.log_metric("mae", mae)
        candidates.append((mae, name, estimator))
        print(f"{name} validation MAE: {mae:.6f} GB")

    validation_mae, selected_name, selected_estimator = min(candidates, key=lambda candidate: candidate[0])
    v1 = make_pipeline(selected_estimator)
    with mlflow.start_run(run_name="v1-refit-and-register") as run:
        v1_run_id = run.info.run_id
        log_training_parameters(data_sha256, v1, fit_rows=len(train), check_rows=len(later_check))
        v1.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
        v1_mae = score(v1, later_check)
        mlflow.log_metrics({"mae": v1_mae, "selected_setting_validation_mae": validation_mae})
        v1_version = register_model(v1, expected_version=1)
    client.set_registered_model_alias(MODEL_NAME, "champion", v1_version)

    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    retraining = pd.concat([train[columns], later_fit[columns]], ignore_index=True)
    v2 = make_pipeline(selected_estimator)
    with mlflow.start_run(run_name="v2-refit-and-register"):
        log_training_parameters(
            data_sha256, v2, fit_rows=len(retraining),
            later_rows_added=len(later_fit), check_rows=len(later_check),
        )
        v2.fit(retraining[FEATURE_COLUMNS], retraining[TARGET_COLUMN])
        v2_mae = score(v2, later_check)
        mlflow.log_metrics({"mae": v2_mae, "v1_mae_on_same_rows": v1_mae})
        v2_version = register_model(v2, expected_version=2)

    promote = v2_mae <= PROMOTION_RATIO * v1_mae
    if promote:
        client.set_registered_model_alias(MODEL_NAME, "champion", v2_version)
    champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    champion = v2 if promote else v1
    temporary_model = MODEL_DIR / "model.joblib.tmp"
    joblib.dump(champion, temporary_model, compress=3)
    temporary_model.replace(MODEL_DIR / "model.joblib")
    update_tracking_findings({
        "experiment_name": EXPERIMENT_NAME,
        "best_run_id": v1_run_id,
        "registered_model_name": MODEL_NAME,
        "champion_version": int(champion_version),
    })
    use_relative_artifact_paths()
    print(f"Selected {selected_name}; v1 run {v1_run_id}")
    print(f"Later split: {len(later_fit)} fit rows, {len(later_check)} shared check rows")
    print(f"v1 MAE: {v1_mae:.6f} GB; v2 MAE: {v2_mae:.6f} GB")
    print(f"Decision: {'promote v2' if promote else 'keep v1'}; champion version {champion_version}")


if __name__ == "__main__":
    main()
