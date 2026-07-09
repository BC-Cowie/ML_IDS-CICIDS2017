"""
src/models/random_forest.py
Random Forest classifier with optional GridSearchCV tuning.
Primary metric: recall / F1 (binary IDS focus).
"""

import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def build(params: dict = None) -> RandomForestClassifier:
    params = params or config.RF_PARAMS
    return RandomForestClassifier(**params)


def train(X_train: np.ndarray, y_train: np.ndarray,
          tune: bool = False) -> RandomForestClassifier:
    print("[random_forest] Training ...")

    if tune:
        print("[random_forest] Running GridSearchCV ...")
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                             random_state=config.RANDOM_STATE)
        base = RandomForestClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )
        gs = GridSearchCV(
            base,
            config.RF_GRID,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X_train, y_train)
        print(f"[random_forest] Best params: {gs.best_params_}")
        model = gs.best_estimator_
    else:
        model = build()
        model.fit(X_train, y_train)

    print("[random_forest] Training complete.")
    return model


def save(model: RandomForestClassifier, path: str = None):
    path = path or os.path.join(config.MODEL_DIR, "random_forest.pkl")
    joblib.dump(model, path)
    print(f"[random_forest] Model saved to {path}")


def load(path: str = None) -> RandomForestClassifier:
    path = path or os.path.join(config.MODEL_DIR, "random_forest.pkl")
    return joblib.load(path)


def feature_importances(model: RandomForestClassifier,
                         feature_names: list) -> list:
    """Returns sorted (feature, importance) tuples descending."""
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances),
                   key=lambda x: x[1], reverse=True)
    return pairs
