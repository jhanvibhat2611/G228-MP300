"""Synthetic tests for monitoring calculations; no project CSVs are needed."""

import sqlite3

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from mp300.monitor import (
    date_split,
    load_registered_model,
    psi_details,
    validate_prediction_output,
)


def test_psi_collapses_repeated_quantiles_and_counts_out_of_range_values():
    reference = pd.Series([3.0] * 20)
    current = pd.Series([-10.0, *([3.0] * 18), 10.0])

    value, effective_bins = psi_details(reference, current, bins=10, smoothing=0.5)

    assert effective_bins == 2  # repeated cuts collapse; outer bins still catch drift
    assert np.isfinite(value)
    assert value > 0


def test_date_split_uses_date_id_then_original_row_order():
    later = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01"],
            "record_id": ["A", "B", "A", "A", "C"],
            "source_marker": ["row-0", "row-1", "row-2", "row-3", "row-4"],
        }
    )

    training, check = date_split(later, train_fraction=0.6)

    assert training.source_marker.tolist() == ["row-2", "row-3", "row-1"]
    assert check.source_marker.tolist() == ["row-4", "row-0"]


def test_prediction_output_checks_columns_order_and_finite_values():
    valid = pd.DataFrame({"record_id": ["R1", "R2"], "prediction": [1.5, 2.5]})
    validate_prediction_output(valid, ["R1", "R2"], expected_rows=2)

    with pytest.raises(ValueError, match="columns"):
        validate_prediction_output(valid[["prediction", "record_id"]], ["R1", "R2"], 2)
    with pytest.raises(ValueError, match="order"):
        validate_prediction_output(valid.iloc[::-1], ["R1", "R2"], 2)
    invalid = valid.copy()
    invalid.loc[1, "prediction"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_prediction_output(invalid, ["R1", "R2"], 2)


def test_registry_resolves_specific_version_and_champion_alias(tmp_path):
    v1_artifact = tmp_path / "models" / "v1" / "artifacts"
    artifact = tmp_path / "models" / "v2" / "artifacts"
    v1_artifact.mkdir(parents=True)
    artifact.mkdir(parents=True)
    model = DummyRegressor(strategy="constant", constant=2.5).fit([[0], [1]], [2.5, 2.5])
    joblib.dump(model, v1_artifact / "model.pkl")
    joblib.dump(model, artifact / "model.pkl")

    database = tmp_path / "mlflow.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE model_versions "
            "(name TEXT, version INTEGER, status TEXT, storage_location TEXT, run_id TEXT)"
        )
        connection.execute(
            "CREATE TABLE registered_model_aliases (alias TEXT, version INTEGER, name TEXT)"
        )
        connection.executemany(
            "INSERT INTO model_versions VALUES (?, ?, ?, ?, ?)",
            [
                ("mp300_gb_per_day", 1, "READY", "models/v1/artifacts", "run-v1"),
                ("mp300_gb_per_day", 2, "READY", "models/v2/artifacts", "run-v2"),
            ],
        )
        connection.execute(
            "INSERT INTO registered_model_aliases VALUES (?, ?, ?)",
            ("champion", 2, "mp300_gb_per_day"),
        )

    v1 = load_registered_model(database, version=1)
    champion = load_registered_model(database, alias="champion")

    assert v1.version == 1 and v1.run_id == "run-v1"
    assert champion.version == 2 and champion.run_id == "run-v2"
    assert champion.model.predict([[3]])[0] == 2.5
