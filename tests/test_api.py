import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src import app as app_module
from src import config
from src.features import build_pipeline

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
          "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}


@pytest.fixture
def client(monkeypatch):
    rng = np.random.default_rng(1)
    X = pd.DataFrame([{**SAMPLE, "age": int(a), "chol": int(c)}
                      for a, c in zip(rng.integers(30, 80, 60), rng.integers(150, 350, 60))])[config.FEATURES]
    y = (X["age"] > 55).astype(int)
    model = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    monkeypatch.setattr(app_module, "_model", model)
    return TestClient(app_module.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1) and 0 <= body["confidence"] <= 1


def test_predict_validation(client):
    assert client.post("/predict", json={**SAMPLE, "sex": 5}).status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json=SAMPLE)
    assert "api_requests_total" in client.get("/metrics").text
