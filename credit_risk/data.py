"""Loading and preprocessing for the UCI 'Default of Credit Card Clients' dataset.

Yeh, I. C. & Lien, C. H. (2009). The comparisons of data mining techniques for the
predictive accuracy of probability of default of credit card clients.
Expert Systems with Applications, 36(2), 2473-2480.
Dataset: https://archive.ics.uci.edu/dataset/350 (CC BY 4.0)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TARGET = "default"

# Official UCI column order (X1..X23, Y)
UCI_COLUMNS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
AMOUNT_COLS = ["LIMIT_BAL"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept the UCI .xls/.csv, the Kaggle 'UCI_Credit_Card.csv' or the ucimlrepo frame."""
    df = df.copy()
    if list(df.columns[:2]) == ["X1", "X2"] or "X1" in df.columns:
        mapping = {f"X{i + 1}": name for i, name in enumerate(UCI_COLUMNS)}
        df = df.rename(columns=mapping)
    target_cols = [c for c in df.columns if "default" in str(c).lower() or str(c) == "Y"]
    if not target_cols:
        raise ValueError("Could not find the default target column.")
    df = df.rename(columns={target_cols[0]: TARGET})
    df = df.drop(columns=[c for c in df.columns if str(c).upper() == "ID"], errors="ignore")
    missing = set(UCI_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df[UCI_COLUMNS + [TARGET]].astype(float)


def load_raw() -> pd.DataFrame:
    """Load from data/raw/ if present, otherwise download via `ucimlrepo`."""
    for path in sorted(RAW_DIR.glob("*")):
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            if "ID" not in df.columns and "X1" not in df.columns:
                df = pd.read_csv(path, header=1)
            return _standardise_columns(df)
        if path.suffix in {".xls", ".xlsx"}:
            return _standardise_columns(pd.read_excel(path, header=1))

    from ucimlrepo import fetch_ucirepo  # optional dependency, only needed for download

    repo = fetch_ucirepo(id=350)
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "default_of_credit_card_clients.csv", index=False)
    return _standardise_columns(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fold undocumented category codes into 'other', as is standard for this dataset."""
    df = df.copy()
    df.loc[~df["EDUCATION"].isin([1, 2, 3]), "EDUCATION"] = 4  # 0, 5, 6 -> other
    df.loc[~df["MARRIAGE"].isin([1, 2]), "MARRIAGE"] = 3  # 0 -> other
    return df


def featurise(df: pd.DataFrame) -> pd.DataFrame:
    """Signed log transform for monetary columns, one-hot for categoricals,
    plus two simple credit-behaviour features."""
    out = df.drop(columns=[TARGET]).copy()
    out["UTILISATION_1"] = (df["BILL_AMT1"] / df["LIMIT_BAL"]).clip(-1, 5)
    out["REPAY_RATIO_1"] = (df["PAY_AMT1"] / df["BILL_AMT1"].clip(lower=1)).clip(0, 5)
    for c in AMOUNT_COLS:
        out[c] = np.sign(out[c]) * np.log1p(np.abs(out[c]))
    out = pd.get_dummies(out, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS, dtype=float)
    return out


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def make_split(seed: int, val_size: float = 0.2, test_size: float = 0.2) -> Split:
    """Stratified train/val/test split. The scaler is fit on the training set only."""
    df = clean(load_raw())
    X = featurise(df)
    y = df[TARGET].to_numpy(dtype=np.float32)
    names = list(X.columns)
    X = X.to_numpy(dtype=np.float32)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size / (1 - test_size), stratify=y_tmp, random_state=seed
    )
    scaler = StandardScaler().fit(X_train)
    f = lambda a: scaler.transform(a).astype(np.float32)  # noqa: E731
    return Split(f(X_train), y_train, f(X_val), y_val, f(X_test), y_test, names)
