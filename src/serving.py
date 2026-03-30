from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response
from fastapi.responses import JSONResponse

from .model_training import ARTIFACTS_DIR, train_and_persist_model


class WineFeatures(BaseModel):
    alcohol: float = Field(..., ge=0.0)
    malic_acid: float = Field(..., ge=0.0)
    ash: float = Field(..., ge=0.0)
    alcalinity_of_ash: float = Field(..., ge=0.0)
    magnesium: float = Field(..., ge=0.0)
    total_phenols: float = Field(..., ge=0.0)
    flavanoids: float = Field(..., ge=0.0)
    nonflavanoid_phenols: float = Field(..., ge=0.0)
    proanthocyanins: float = Field(..., ge=0.0)
    color_intensity: float = Field(..., ge=0.0)
    hue: float = Field(..., ge=0.0)
    od280_od315_of_diluted_wines: float = Field(..., ge=0.0)
    proline: float = Field(..., ge=0.0)


REQUEST_COUNT = Counter(
    "model_inference_requests_total",
    "Total number of inference requests.",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Latency of inference requests in seconds.",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTED_CLASS = Counter(
    "model_predictions_total",
    "Predicted class distribution.",
    ["predicted_class"],
)
PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Confidence distribution for model predictions.",
    buckets=(0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
MODEL_INFO = Gauge(
    "model_metadata",
    "Static metadata for the served model.",
    ["model_name", "dataset_name"],
)


MODEL_LOCK = Lock()
MODEL = None
MODEL_PATH = ARTIFACTS_DIR / "wine_classifier.joblib"


def _load_model():
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            if not MODEL_PATH.exists():
                train_and_persist_model()
            MODEL = joblib.load(MODEL_PATH)
            MODEL_INFO.labels(model_name="logistic_regression", dataset_name="wine").set(1)
    return MODEL


app = FastAPI(title="ML Model Serving Observability", version="1.0.0")


@app.on_event("startup")
def startup_event() -> None:
    _load_model()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    endpoint = request.url.path
    REQUEST_COUNT.labels(endpoint=endpoint, status="validation_error").inc()
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "error_type": "validation_error"})


@app.get("/health")
def health() -> dict:
    model_path = str(MODEL_PATH)
    return {"status": "ok", "model_loaded": Path(model_path).exists(), "model_path": model_path}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: WineFeatures) -> dict:
    model = _load_model()
    started_at = time.perf_counter()
    endpoint = "/predict"
    try:
        frame = pd.DataFrame([payload.model_dump()]).rename(
            columns={"od280_od315_of_diluted_wines": "od280/od315_of_diluted_wines"}
        )
        probabilities = model.predict_proba(frame)[0]
        predicted_class = int(probabilities.argmax())
        confidence = float(probabilities[predicted_class])
        elapsed = time.perf_counter() - started_at

        REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        PREDICTED_CLASS.labels(predicted_class=str(predicted_class)).inc()
        PREDICTION_CONFIDENCE.observe(confidence)

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "class_probabilities": [round(float(value), 4) for value in probabilities],
            "latency_ms": round(elapsed * 1000, 2),
        }
    except Exception as exc:  # pragma: no cover - defensive path
        REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        elapsed = time.perf_counter() - started_at
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        raise HTTPException(status_code=500, detail=f"prediction_failed: {exc}") from exc
