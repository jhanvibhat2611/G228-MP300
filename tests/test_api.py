"""Exercise the API with a temporary joblib artifact and no raw CSVs."""

import io
import json
from contextlib import redirect_stdout

import joblib
import pytest
from fastapi.testclient import TestClient

from app import main

PAYLOAD = {
    "circle": "Semi-urban",
    "phone_tier": "mid",
    "wifi_hours": 3.0,
    "video_minutes": 0,
    "plan_gb_limit": 8,
    "apps_installed": 32,
    "screen_hours": 2.5,
    "age": 24,
}


class SamplePredictor:
    def predict(self, features):
        assert features.columns.tolist() == [
            "age", "screen_hours", "apps_installed", "plan_gb_limit",
            "video_minutes", "wifi_hours", "phone_tier", "circle",
        ]
        assert features.to_dict(orient="records") == [PAYLOAD]
        return [2.75]


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    monkeypatch.setattr(main, "MODEL_PATH", path)
    main.load_model.cache_clear()
    yield path
    main.load_model.cache_clear()


@pytest.fixture
def client(model_path):
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_without_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_at_valid_boundaries(model_path):
    joblib.dump(SamplePredictor(), model_path)
    stdout = io.StringIO()
    with redirect_stdout(stdout), TestClient(main.app) as client:
        response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {"prediction": 2.75}
    log = json.loads(stdout.getvalue())
    assert log.keys() == {"event", "status", "latency_ms"}
    assert log["event"] == "prediction"
    assert log["status"] == "ok"
    assert log["latency_ms"] >= 0


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("plan_gb_limit", 8.01, "less_than_equal"),
        ("video_minutes", -1, "greater_than_equal"),
        ("record_id", "test-id", "extra_forbidden"),
    ],
)
def test_predict_rejects_invalid_input(client, field, value, error_type):
    response = client.post("/predict", json={**PAYLOAD, field: value})

    assert response.status_code == 422
    error, = response.json()["detail"]
    assert error["loc"] == ["body", field]
    assert error["type"] == error_type


def test_predict_requires_all_fields(client):
    payload = PAYLOAD.copy()
    del payload["circle"]
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    error, = response.json()["detail"]
    assert error["loc"] == ["body", "circle"]
    assert error["type"] == "missing"


def test_predict_recovers_when_model_is_added(client, model_path):
    response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction model is unavailable"}

    joblib.dump(SamplePredictor(), model_path)
    assert client.post("/predict", json=PAYLOAD).json() == {"prediction": 2.75}
