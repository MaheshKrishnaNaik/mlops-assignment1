import pandas as pd

from src import config
from src.data import clean_data


def raw_df():
    return pd.DataFrame({
        "age": [63, 50, 50], "sex": [1, 0, 0], "cp": [1, 2, 2], "trestbps": [145, None, None],
        "chol": [233, 200, 200], "fbs": [1, 0, 0], "restecg": [2, 0, 0], "thalach": [150, 160, 160],
        "exang": [0, 1, 1], "oldpeak": [2.3, 1.0, 1.0], "slope": [3, 1, 1], "ca": [0, None, None],
        "thal": [6, 3, 3], "num": [0, 2, 2],
    })


def test_target_binarised_and_no_nulls():
    df = clean_data(raw_df())
    assert set(df[config.TARGET]) <= {0, 1}
    assert "num" not in df.columns
    assert df.isna().sum().sum() == 0


def test_duplicates_removed():
    assert len(clean_data(raw_df())) == 2


def test_question_marks_handled():
    raw = raw_df().astype(object)
    raw.loc[0, "ca"] = "?"
    assert clean_data(raw).isna().sum().sum() == 0
