"""
src/models/ensemble.py
Ensemble methods: stacking, bagging, boosting, and meta-processing.
Loads trained base models and combines them.
"""

import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import (
    StackingClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from src.models import random_forest, xgboost_model, svm


# ── Stacking ─────────────────────────────────────────────────────────────

def build_stacking(base_estimators: list, binary: bool = True):
    """
    base_estimators: list of (name, fitted_sklearn_estimator)
    Meta-learner: Logistic Regression or XGBoost depending on config.
    """
    if config.STACK_META_LEARNER == "logistic":
        meta = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )
    else:
        meta = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=config.RANDOM_STATE,
        )

    cv = StratifiedKFold(
        n_splits=config.STACK_CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        cv=cv,
        passthrough=False,    # only meta-features, not raw features
        n_jobs=-1,
    )
    return stack


def train_stacking(X_train: np.ndarray, y_train: np.ndarray,
                   base_estimators: list, binary: bool = True):
    print("[ensemble] Training stacking classifier ...")
    model = build_stacking(base_estimators, binary=binary)
    model.fit(X_train, y_train)
    print("[ensemble] Stacking training complete.")
    return model


# ── Voting ────────────────────────────────────────────────────────────────

def train_voting(X_train: np.ndarray, y_train: np.ndarray,
                 base_estimators: list, voting: str = "soft"):
    """
    Soft voting averages predicted probabilities.
    Hard voting uses majority class vote.
    """
    print(f"[ensemble] Training voting classifier ({voting}) ...")
    model = VotingClassifier(estimators=base_estimators, voting=voting, n_jobs=-1)
    model.fit(X_train, y_train)
    print("[ensemble] Voting training complete.")
    return model


# ── Bagging ───────────────────────────────────────────────────────────────

def train_bagging(X_train: np.ndarray, y_train: np.ndarray,
                  base_estimator=None, n_estimators: int = 20):
    """
    Bagging over a base estimator (defaults to RF).
    """
    print("[ensemble] Training bagging classifier ...")
    if base_estimator is None:
        from sklearn.tree import DecisionTreeClassifier
        base_estimator = DecisionTreeClassifier(max_depth=10)

    model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[ensemble] Bagging training complete.")
    return model


# ── Gradient Boosting ─────────────────────────────────────────────────────

def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray):
    print("[ensemble] Training gradient boosting classifier ...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print("[ensemble] Gradient boosting training complete.")
    return model


# ── Meta-processing (predict from saved base models) ──────────────────────

class MetaProcessor:
    """
    Loads all saved base models and combines their predictions
    without retraining — useful for GUI inference.
    """

    def __init__(self, binary: bool = True):
        self.binary  = binary
        self.models  = {}
        self._load_all()

    def _load_all(self):
        try:
            self.models["rf"]  = random_forest.load()
            print("[meta] Random Forest loaded")
        except Exception as e:
            print(f"[meta] RF not found: {e}")

        try:
            self.models["xgb"] = xgboost_model.load()
            print("[meta] XGBoost loaded")
        except Exception as e:
            print(f"[meta] XGBoost not found: {e}")

        try:
            self.models["svm"] = svm.load()
            print("[meta] SVM loaded")
        except Exception as e:
            print(f"[meta] SVM not found: {e}")

        try:
            from src.models.neural_network import NeuralNetworkModel
            self.models["nn"] = NeuralNetworkModel.load()
            print("[meta] Neural Network loaded")
        except Exception as e:
            print(f"[meta] NN not found: {e}")

    def predict(self, X: np.ndarray, strategy: str = "majority") -> np.ndarray:
        """
        strategy:
            "majority"  — hard vote across all loaded models
            "any"       — flag as attack if ANY model says attack (max recall)
            "all"       — flag as attack only if ALL models agree (max precision)
        """
        if not self.models:
            raise RuntimeError("No models loaded.")

        preds = np.stack(
            [m.predict(X) for m in self.models.values()], axis=1
        )

        if strategy == "majority":
            return (preds.sum(axis=1) > preds.shape[1] / 2).astype(int)
        elif strategy == "any":
            return (preds.sum(axis=1) >= 1).astype(int)
        elif strategy == "all":
            return (preds.sum(axis=1) == preds.shape[1]).astype(int)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def predict_proba_avg(self, X: np.ndarray) -> np.ndarray:
        """Average predicted probabilities across models that support it."""
        probas = []
        for name, m in self.models.items():
            if hasattr(m, "predict_proba"):
                probas.append(m.predict_proba(X))
        if not probas:
            raise RuntimeError("No models support predict_proba.")
        return np.mean(probas, axis=0)


# ── Save / Load ────────────────────────────────────────────────────────────

def save(model, name: str):
    path = os.path.join(config.MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"[ensemble] Model saved to {path}")


def load(name: str):
    path = os.path.join(config.MODEL_DIR, f"{name}.pkl")
    return joblib.load(path)
