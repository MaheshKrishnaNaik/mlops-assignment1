#!/usr/bin/env bash
# Full setup + run on a Linux VM. Usage: bash scripts/vm_setup.sh   (run from the repo root)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
HOST_PORT="${HOST_PORT:-8001}"
SAMPLE='{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'

echo "== 1. Python env =="
python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt

echo "== 2. Data, EDA, tests, lint, training =="
python -m src.data
PYTHONPATH=. python notebooks/eda.py
flake8 src tests
pytest -q
rm -rf mlruns
python -m src.train

echo "== 3. Docker build + run =="
docker build -t heart-api:latest .
docker rm -f heart-api >/dev/null 2>&1 || true
docker run -d --name heart-api -p "${HOST_PORT}:8000" heart-api:latest
sleep 6
curl -fsS "http://localhost:${HOST_PORT}/health"; echo
curl -fsS -X POST "http://localhost:${HOST_PORT}/predict" -H "Content-Type: application/json" -d "$SAMPLE"; echo
docker rm -f heart-api

echo "== 4. Kubernetes (Minikube) =="
mkdir -p "$HOME/.local/bin"
if ! command -v kubectl >/dev/null; then
  curl -fL -o "$HOME/.local/bin/kubectl" https://dl.k8s.io/release/v1.31.0/bin/linux/amd64/kubectl
  chmod +x "$HOME/.local/bin/kubectl"
fi
if ! command -v minikube >/dev/null; then
  curl -fL -o "$HOME/.local/bin/minikube" https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
  chmod +x "$HOME/.local/bin/minikube"
fi
minikube status >/dev/null 2>&1 || minikube start --driver=docker
minikube image load heart-api:latest
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/heart-api --timeout=180s
kubectl get pods,svc

echo "== 5. Test through the cluster =="
kubectl port-forward svc/heart-api "${HOST_PORT}:80" >/tmp/pf.log 2>&1 &
PF=$!
sleep 5
curl -fsS "http://localhost:${HOST_PORT}/health"; echo
curl -fsS -X POST "http://localhost:${HOST_PORT}/predict" -H "Content-Type: application/json" -d "$SAMPLE"; echo
kill $PF

echo "Done. Take screenshots now, then: python scripts/make_report.py"
