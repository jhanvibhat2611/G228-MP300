"""Serve the saved MP300 model through FastAPI."""

import json
from functools import cache
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.mp300.clean import FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

app = FastAPI(title="MP300 Mobile Data Usage API")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    age: int
    screen_hours: float
    apps_installed: int
    plan_gb_limit: float = Field(le=8)
    video_minutes: float = Field(ge=0)
    wifi_hours: float
    phone_tier: str
    circle: str


@cache
def load_model():
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    started_at = perf_counter()
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Prediction model is unavailable") from exc

    features = pd.DataFrame([request.model_dump()], columns=FEATURE_COLUMNS)
    prediction = float(model.predict(features)[0])

    event = {
        "event": "prediction",
        "status": "ok",
        "latency_ms": round((perf_counter() - started_at) * 1000, 2),
    }
    print(json.dumps(event), flush=True)
    return {"prediction": prediction}
