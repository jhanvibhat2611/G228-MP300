"""Serve the saved MP300 model through FastAPI."""

import json
import logging
import sys
from contextlib import asynccontextmanager
from functools import cache
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.mp300.clean import FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
prediction_logger = logging.getLogger("mp300.prediction")
prediction_logger.setLevel(logging.INFO)
prediction_logger.propagate = False


class PredictionJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "event": record.getMessage(),
            "status": record.status,
            "latency_ms": record.latency_ms,
        })


@asynccontextmanager
async def lifespan(app: FastAPI):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(PredictionJsonFormatter())
    prediction_logger.addHandler(handler)
    try:
        yield
    finally:
        prediction_logger.removeHandler(handler)
        handler.close()


app = FastAPI(title="MP300 Mobile Data Usage API", lifespan=lifespan)


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

    prediction_logger.info("prediction", extra={
        "status": "ok",
        "latency_ms": round((perf_counter() - started_at) * 1000, 2),
    })
    return {"prediction": prediction}
