"""
src/models/xgboost_model.py
XGBoost classifier with optional GridSearchCV tuning.
Handles binary (scale_pos_weight) and multiclass automatically.
"""

import os
import sys
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def build(params: dict = None, binary: bool = True,
          scale_pos_weight: float = None) -> XGBClassifier:
    params = dict(config.XGB_PARAMS)  # copy so we don't mutate config
    if binary:
        params["objective"] = "binary:logistic"
        if scale_pos_weight:
            params["scale_pos_weight"] = scale_pos_weight
    else:
        params["objective"] = "multi:softprob"

    return XGBClassifier(**params)


def train(X_train: np.ndarray, y_train: np.ndarray,
          binary: bool = True, tune: bool = False) -> XGBClassifier:
    print("[xgboost] Training ...")

    # Compute scale_pos_weight for binary imbalance
    spw = None
    if binary:
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"[xgboost] scale_pos_weight = {spw:.2f}")

    if tune:
        print("[xgboost] Running GridSearchCV ...")
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                             random_state=config.RANDOM_STATE)
        base = build(binary=binary, scale_pos_weight=spw)
        gs = GridSearchCV(
            base,
            config.XGB_GRID,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X_train, y_train)
        print(f"[xgboost] Best params: {gs.best_params_}")
        model = gs.best_estimator_
    else:
        model = build(binary=binary, scale_pos_weight=spw)
        model.fit(X_train, y_train)

    print("[xgboost] Training complete.")
    return model


def save(model: XGBClassifier, path: str = None):
    path = path or os.path.join(config.MODEL_DIR, "xgboost.pkl")
    joblib.dump(model, path)
    print(f"[xgboost] Model saved to {path}")


def load(path: str = None) -> XGBClassifier:
    path = path or os.path.join(config.MODEL_DIR, "xgboost.pkl")
    return joblib.load(path)


def feature_importances(model: XGBClassifier,
                         feature_names: list) -> list:
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances),
                   key=lambda x: x[1], reverse=True)
    return pairs
