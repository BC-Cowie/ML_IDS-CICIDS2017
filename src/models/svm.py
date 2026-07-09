"""
src/models/svm.py
SVM classifier with optional GridSearchCV tuning.
Note: SVM is slow on large datasets — consider subsampling for tuning.
"""

import os
import sys
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

# SVM gets slow past ~100k samples — cap training size
SVM_MAX_SAMPLES = 50_000


def build(params: dict = None) -> SVC:
    params = params or config.SVM_PARAMS
    return SVC(**params)


def train(X_train: np.ndarray, y_train: np.ndarray,
          tune: bool = False) -> SVC:
    print("[svm] Training ...")

    # Subsample if dataset is too large
    if X_train.shape[0] > SVM_MAX_SAMPLES:
        print(f"[svm] Dataset too large for SVM — subsampling to {SVM_MAX_SAMPLES:,}")
        rng = np.random.default_rng(config.RANDOM_STATE)
        idx = rng.choice(X_train.shape[0], SVM_MAX_SAMPLES, replace=False)
        X_tr = X_train[idx]
        y_tr = y_train[idx]
    else:
        X_tr, y_tr = X_train, y_train

    if tune:
        print("[svm] Running GridSearchCV (this may take a while) ...")
        cv = StratifiedKFold(n_splits=3, shuffle=True,
                             random_state=config.RANDOM_STATE)
        base = SVC(class_weight="balanced", probability=True,
                   random_state=config.RANDOM_STATE)
        gs = GridSearchCV(
            base,
            config.SVM_GRID,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X_tr, y_tr)
        print(f"[svm] Best params: {gs.best_params_}")
        model = gs.best_estimator_
    else:
        model = build()
        model.fit(X_tr, y_tr)

    print("[svm] Training complete.")
    return model


def save(model: SVC, path: str = None):
    path = path or os.path.join(config.MODEL_DIR, "svm.pkl")
    joblib.dump(model, path)
    print(f"[svm] Model saved to {path}")


def load(path: str = None) -> SVC:
    path = path or os.path.join(config.MODEL_DIR, "svm.pkl")
    return joblib.load(path)
