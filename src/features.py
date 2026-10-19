from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL),
    ])


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])
