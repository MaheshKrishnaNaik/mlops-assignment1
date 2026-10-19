# MLOps Assignment 1 - Heart Disease Risk Prediction

Author: 2025af05162 (2025af05162@wilp.bits-pilani.ac.in)
Repository: https://github.com/MaheshKrishnaNaik/mlops-assignment1

## Architecture

```
UCI data -> src/data.py (clean) -> src/train.py (LogReg + RandomForest, GridSearch, 5-fold CV)
         -> MLflow (params/metrics/plots/models) -> models/model.joblib (full sklearn Pipeline)
         -> FastAPI (/predict, /health, /metrics) -> Docker -> Kubernetes (LoadBalancer/Ingress)
         -> Prometheus -> Grafana
GitHub Actions: flake8 -> pytest -> data -> train -> artifacts -> docker build + smoke test
```

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m src.data                  # downloads UCI data (data/raw) and writes data/processed/heart_disease_clean.csv
PYTHONPATH=. python notebooks/eda.py   # EDA plots -> screenshots/eda/
python -m src.train                 # trains, logs to MLflow, saves models/model.joblib
mlflow ui                           # view runs at http://127.0.0.1:5000
pytest -v && flake8 src tests
```

## Data and EDA
Heart Disease UCI (Cleveland, 303 rows, 13 features). `?`/missing values: median imputation for numeric, mode for
categorical (`ca`, `thal` have a few missing). Target `num` is binarised (>0 = disease). Class balance is ~54% / 46%,
so accuracy is a fair metric and no resampling is needed. Plots: `screenshots/eda/`.

## Modelling
Preprocessing (StandardScaler on numeric, OneHotEncoder on categorical) is part of one sklearn `Pipeline`, so
training and serving use identical transforms. Two models are tuned with GridSearchCV (ROC-AUC, stratified 5-fold):
Logistic Regression (C) and Random Forest (n_estimators, max_depth). Best model is picked by CV ROC-AUC.

| Model | CV acc | CV prec | CV recall | CV ROC-AUC | Test ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (C=1) | 0.847 | 0.877 | 0.783 | 0.902 | 0.966 |
| Random Forest (100, depth None) | 0.818 | 0.817 | 0.783 | 0.897 | 0.943 |

Selected: Logistic Regression (higher CV scores, simpler, better calibrated for a risk probability).
Test numbers come from a single 61-row hold-out split, so treat them as noisy; CV is the main comparison.

## Experiment tracking
MLflow experiment `heart-disease`: one run per model with best params, CV and test metrics, ROC curve and
confusion-matrix plots, and the fitted pipeline. Add screenshots of `mlflow ui` to `screenshots/`.

## API and Docker
```bash
uvicorn src.app:app --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'

docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest
```
Response: `{"prediction":0,"confidence":0.83,"risk_probability":0.17}`. Every request is logged and exposed as
Prometheus metrics at `/metrics`.

## Kubernetes (Docker Desktop or Minikube)
```bash
docker build -t heart-api:latest .
# Minikube only: minikube image load heart-api:latest
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/ingress.yaml        # optional, needs an ingress controller
kubectl get svc heart-api                # Docker Desktop: http://localhost ; Minikube: minikube tunnel
```

## Monitoring
```bash
cd monitoring && docker compose up -d     # Prometheus :9090, Grafana :3000 (add Prometheus data source http://prometheus:9090)
```
Useful queries: `rate(api_requests_total[1m])`, `histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[1m]))`,
`api_predictions_total`.

## CI/CD
`.github/workflows/ci.yml` runs lint, tests, data prep, training, uploads artifacts (test results, model, MLflow
runs), builds the image and smoke-tests the container. The job fails on lint or test errors.

## Still to do manually (needs Docker/GitHub on your machine)
Push to your own GitHub repo, capture screenshots (Actions run, Docker run, kubectl, Grafana, MLflow) into
`screenshots/`, record the short video, and export this README to a PDF report.
