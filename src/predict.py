"""
src/predict.py
Inference on new data using saved models.
Used by the GUI backend.
"""

import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from src.preprocess import load_scaler


ATTACK_LABEL = {0: "BENIGN", 1: "ATTACK"}


def load_model(model_name: str):
    """
    model_name: "random_forest" | "xgboost" | "svm" | "neural_network"
    """
    if model_name == "neural_network":
        from src.models.neural_network import NeuralNetworkModel
        return NeuralNetworkModel.load()

    path = os.path.join(config.MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def predict_binary(X_raw: np.ndarray, model_name: str = "random_forest",
                   threshold: float = 0.5) -> dict:
    """
    Predict binary (BENIGN / ATTACK) on raw (unscaled) feature matrix.
    Returns dict with predictions and confidence scores.
    threshold: lower = higher recall (fewer false negatives) — default 0.5
    """
    scaler = load_scaler()
    X = scaler.transform(X_raw)
    model = load_model(model_name)

    if hasattr(model, "predict_proba"):
        proba  = model.predict_proba(X)
        scores = proba[:, 1]  # probability of ATTACK
        preds  = (scores >= threshold).astype(int)
    else:
        preds  = model.predict(X)
        scores = preds.astype(float)

    return {
        "predictions": preds,
        "labels":      [ATTACK_LABEL[p] for p in preds],
        "confidence":  scores,
    }


def predict_multiclass(X_raw: np.ndarray,
                        model_name: str = "random_forest",
                        label_encoder=None) -> dict:
    """
    Predict attack class on raw feature matrix.
    label_encoder: sklearn LabelEncoder fitted during preprocessing.
    """
    scaler = load_scaler()
    X = scaler.transform(X_raw)
    model = load_model(model_name)
    preds = model.predict(X)

    if label_encoder:
        class_names = label_encoder.inverse_transform(preds)
    else:
        class_names = [str(p) for p in preds]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

    return {
        "predictions":  preds,
        "class_names":  class_names,
        "probabilities": proba,
    }


def predict_single_flow(flow_dict: dict, feature_names: list,
                         model_name: str = "random_forest",
                         threshold: float = 0.5) -> dict:
    """
    Predict a single network flow from a dict of feature values.
    Used by the GUI for manual input / live demo.
    """
    X_raw = np.array([[flow_dict.get(f, 0.0) for f in feature_names]])
    return predict_binary(X_raw, model_name=model_name, threshold=threshold)


def ensemble_predict(X_raw: np.ndarray,
                     strategy: str = "majority",
                     threshold: float = 0.5) -> dict:
    """
    Run MetaProcessor ensemble prediction.
    strategy: "majority" | "any" | "all"
    """
    from src.models.ensemble import MetaProcessor
    scaler = load_scaler()
    X = scaler.transform(X_raw)
    meta = MetaProcessor(binary=True)
    preds = meta.predict(X, strategy=strategy)

    try:
        proba  = meta.predict_proba_avg(X)
        scores = proba[:, 1]
    except Exception:
        scores = preds.astype(float)

    return {
        "predictions": preds,
        "labels":      [ATTACK_LABEL[p] for p in preds],
        "confidence":  scores,
        "strategy":    strategy,
    }
