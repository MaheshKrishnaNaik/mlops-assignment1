"""Train, cross-validate and track models with MLflow."""
import json
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_validate,  # noqa: E402
                                     train_test_split)

from src import config  # noqa: E402
from src.data import load_clean  # noqa: E402
from src.features import build_pipeline  # noqa: E402

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=config.RANDOM_STATE),
        {"model__n_estimators": [100, 300], "model__max_depth": [3, 5, None]},
    ),
}
SCORING = ["accuracy", "precision", "recall", "roc_auc"]


def evaluate(pipe, X, y):
    pred, proba = pipe.predict(X), pipe.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def train_all(df=None, experiment="heart-disease"):
    df = load_clean() if df is None else df
    X, y = df[config.FEATURES], df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    mlflow.set_experiment(experiment)
    best_name, best_pipe, best_auc, results = None, None, -1.0, {}

    for name, (est, grid) in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(build_pipeline(est), grid, cv=cv, scoring="roc_auc")
            search.fit(X_tr, y_tr)
            pipe = search.best_estimator_
            cvs = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=SCORING)
            cv_metrics = {f"cv_{m}": float(cvs[f"test_{m}"].mean()) for m in SCORING}
            test_metrics = {f"test_{k}": v for k, v in evaluate(pipe, X_te, y_te).items()}
            mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
            mlflow.log_metrics({**cv_metrics, **test_metrics})
            with tempfile.TemporaryDirectory() as tmp:
                plots = (("roc_curve", RocCurveDisplay), ("confusion_matrix", ConfusionMatrixDisplay))
                for plot, disp in plots:
                    disp.from_estimator(pipe, X_te, y_te)
                    p = Path(tmp) / f"{plot}.png"
                    plt.savefig(p)
                    plt.close()
                    mlflow.log_artifact(str(p))
            mlflow.sklearn.log_model(pipe, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
            best_params = {k: str(v) for k, v in search.best_params_.items()}
            results[name] = {**cv_metrics, **test_metrics, "params": best_params}
            if cv_metrics["cv_roc_auc"] > best_auc:
                best_name, best_pipe, best_auc = name, pipe, cv_metrics["cv_roc_auc"]

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps({"best_model": best_name, "results": results}, indent=2))
    return best_name, results


def main():
    name, results = train_all()
    print(f"Best model: {name}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
