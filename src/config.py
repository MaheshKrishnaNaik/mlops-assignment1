from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "heart_disease_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_disease_clean.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_URL = "https://archive.ics.uci.edu/static/public/45/data.csv"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"
RANDOM_STATE = 42
