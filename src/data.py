"""Data acquisition and cleaning."""
import urllib.request

import pandas as pd

from src import config


def download_data(path=config.RAW_PATH, url=config.DATA_URL):
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values (median/mode) and binarise target (num>0 -> 1)."""
    df = df.copy().apply(pd.to_numeric, errors="coerce")
    if "num" in df.columns:
        df[config.TARGET] = (df.pop("num") > 0).astype(int)
    for col in config.FEATURES:
        if col in config.CATEGORICAL:
            modes = df[col].mode()
            fill = modes.iloc[0] if len(modes) else 0
        else:
            fill = df[col].median()
        df[col] = df[col].fillna(fill)
    return df.drop_duplicates().reset_index(drop=True)


def load_clean(path=config.CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    raw = download_data()
    df = clean_data(pd.read_csv(raw))
    config.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_PATH, index=False)
    print(f"Saved {len(df)} rows to {config.CLEAN_PATH}")


if __name__ == "__main__":
    main()
