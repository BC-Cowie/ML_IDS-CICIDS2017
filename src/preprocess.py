"""
src/preprocess.py
Loads, cleans, and prepares CICIDS2017 data for training.
Produces both binary and multiclass label sets.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data(path: str = config.DATA_PATH) -> pd.DataFrame:
    print(f"[preprocess] Loading data from {path} ...")
    df = pd.read_csv(path, low_memory=False)

    # Strip whitespace from column names and string values
    df.columns = df.columns.str.strip()
    df[config.LABEL_COL] = df[config.LABEL_COL].str.strip()

    # Sample if configured — use during dev/tuning, set None for full run
    if config.SAMPLE_SIZE is not None and len(df) > config.SAMPLE_SIZE:
        print(f"[preprocess] Sampling {config.SAMPLE_SIZE:,} rows from {len(df):,} ...")
        df = df.sample(config.SAMPLE_SIZE, random_state=config.RANDOM_STATE)
        df = df.reset_index(drop=True)

    print(f"[preprocess] Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"[preprocess] Class distribution:\n{df[config.LABEL_COL].value_counts()}\n")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("[preprocess] Cleaning ...")

    before = len(df)
    df = df.drop_duplicates()
    print(f"[preprocess] Dropped {before - len(df):,} duplicate rows")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        print(f"[preprocess] Imputing NaN in {len(nan_cols)} columns")
        df[nan_cols] = df[nan_cols].fillna(df[nan_cols].median())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cap = df[numeric_cols].quantile(config.OUTLIER_CLIP_PERCENTILE / 100)
    df[numeric_cols] = df[numeric_cols].clip(upper=cap, axis=1)

    return df


def get_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c != config.LABEL_COL]


def make_binary_labels(df: pd.DataFrame) -> np.ndarray:
    """0 = BENIGN, 1 = ATTACK"""
    return (df[config.LABEL_COL] != config.BENIGN_LABEL).astype(int).values


def make_multiclass_labels(df: pd.DataFrame):
    le = LabelEncoder()
    y = le.fit_transform(df[config.LABEL_COL])
    return y, le


def split_and_scale(X: np.ndarray, y: np.ndarray,
                    apply_smote: bool = config.APPLY_SMOTE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if apply_smote:
        unique, counts = np.unique(y_train, return_counts=True)
        min_count = counts.min()
        if min_count < 6:
            k = max(1, min_count - 1)
            print(f"[preprocess] SMOTE k_neighbors adjusted to {k}")
            sm = SMOTE(k_neighbors=k, random_state=config.RANDOM_STATE)
        else:
            sm = SMOTE(sampling_strategy=config.SMOTE_STRATEGY,
                       random_state=config.RANDOM_STATE)
        print(f"[preprocess] Applying SMOTE ...")
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"[preprocess] After SMOTE: {X_train.shape[0]:,} training samples")

    return X_train, X_test, y_train, y_test, scaler


def save_scaler(scaler: StandardScaler, path: str = None):
    path = path or os.path.join(config.MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, path)
    print(f"[preprocess] Scaler saved to {path}")


def load_scaler(path: str = None) -> StandardScaler:
    path = path or os.path.join(config.MODEL_DIR, "scaler.pkl")
    return joblib.load(path)


def run(mode: str = "binary"):
    df = load_data()
    df = clean(df)

    feature_names = get_features(df)
    if hasattr(config, 'DROP_FEATURES') and config.DROP_FEATURES:
        feature_names = [f for f in feature_names if f not in config.DROP_FEATURES]
        print(f"[preprocess] Dropped features: {config.DROP_FEATURES}")
    X = df[feature_names].values

    if mode == "binary":
        y = make_binary_labels(df)
        X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
        save_scaler(scaler)
        return X_train, X_test, y_train, y_test, scaler, feature_names

    elif mode == "multiclass":
        y, le = make_multiclass_labels(df)
        X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
        save_scaler(scaler)
        return X_train, X_test, y_train, y_test, scaler, feature_names, le

    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    run("binary")