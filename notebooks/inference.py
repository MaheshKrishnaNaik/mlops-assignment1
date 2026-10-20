# %% [markdown]
# # Inference - load the saved pipeline and predict for one patient
# Run: `PYTHONPATH=. python notebooks/inference.py`

# %%
import joblib
import pandas as pd

from src import config

model = joblib.load(config.MODEL_PATH)
patient = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
           "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
proba = model.predict_proba(pd.DataFrame([patient])[config.FEATURES])[0, 1]
print(f"Disease probability: {proba:.3f} -> {'disease' if proba >= 0.5 else 'no disease'}")
