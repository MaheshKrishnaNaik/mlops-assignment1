import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import build_pipeline
from src.train import CANDIDATES, evaluate


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "age": rng.integers(30, 80, n), "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 350, n), "thalach": rng.integers(90, 200, n),
        "oldpeak": rng.random(n) * 4, "sex": rng.integers(0, 2, n), "cp": rng.integers(1, 5, n),
        "fbs": rng.integers(0, 2, n), "restecg": rng.integers(0, 3, n), "exang": rng.integers(0, 2, n),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n), "thal": rng.choice([3, 6, 7], n),
    })
    y = (df["age"] + df["oldpeak"] * 10 > 70).astype(int)
    return df[config.FEATURES], y


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_pipeline_fits_and_predicts(toy, name):
    X, y = toy
    pipe = build_pipeline(CANDIDATES[name][0]).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_unseen_category_does_not_crash(toy):
    X, y = toy
    pipe = build_pipeline(CANDIDATES["logistic_regression"][0]).fit(X, y)
    X2 = X.head(2).copy()
    X2["cp"] = 99
    assert len(pipe.predict(X2)) == 2


def test_evaluate_returns_metrics(toy):
    X, y = toy
    pipe = build_pipeline(CANDIDATES["random_forest"][0]).fit(X, y)
    m = evaluate(pipe, X, y)
    assert set(m) == {"accuracy", "precision", "recall", "roc_auc"}
    assert all(0 <= v <= 1 for v in m.values())
