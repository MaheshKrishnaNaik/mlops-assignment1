"""FastAPI serving app with request logging and Prometheus metrics."""
import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-api")

app = FastAPI(title="Heart Disease Risk API")
REQS = Counter("api_requests_total", "Total requests", ["path", "status"])
LAT = Histogram("api_request_latency_seconds", "Request latency", ["path"])
PREDS = Counter("api_predictions_total", "Predictions by class", ["label"])
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(config.MODEL_PATH)
    return _model


class Patient(BaseModel):
    age: float = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0)
    chol: float = Field(gt=0)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0)
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: float = Field(ge=3, le=7)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    dur = time.perf_counter() - start
    REQS.labels(request.url.path, response.status_code).inc()
    LAT.labels(request.url.path).observe(dur)
    log.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, dur * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    X = pd.DataFrame([patient.model_dump()])[config.FEATURES]
    proba = float(get_model().predict_proba(X)[0, 1])
    label = int(proba >= 0.5)
    PREDS.labels(str(label)).inc()
    log.info("prediction=%d confidence=%.3f", label, proba)
    return {"prediction": label, "confidence": proba if label else 1 - proba, "risk_probability": proba}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
