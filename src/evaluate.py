"""
src/evaluate.py
Evaluation utilities: metrics, classification reports, cross-validation.
Primary focus: recall and F1 (binary IDS — minimise false negatives).
"""

import os
import sys
import json
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None,
                    binary: bool = True) -> dict:
    avg = "binary" if binary else "weighted"
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
    }

    if y_prob is not None:
        try:
            if binary:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(
                    y_true, y_prob,
                    multi_class="ovr",
                    average="weighted",
                )
            metrics["auc_roc"] = round(auc, 4)
        except Exception:
            metrics["auc_roc"] = None

    return metrics


def print_report(y_true: np.ndarray, y_pred: np.ndarray,
                 target_names: list = None, model_name: str = ""):
    print(f"\n{'='*60}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*60}")
    print(classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
    ))


def get_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def cross_validate_model(model, X: np.ndarray, y: np.ndarray,
                          binary: bool = True) -> dict:
    """
    Run stratified k-fold CV and return mean ± std for key metrics.
    """
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {}
    for key, vals in results.items():
        if key.startswith("test_"):
            metric = key.replace("test_", "")
            summary[metric] = {
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals)), 4),
            }
    return summary


def save_results(results: dict, filename: str = "results.json"):
    path = os.path.join(config.RESULT_DIR, filename)
    # Remove non-serialisable objects (models, arrays)
    clean = {}
    for model_name, data in results.items():
        clean[model_name] = {
            k: v for k, v in data.items()
            if isinstance(v, (int, float, str, dict, list, type(None)))
        }
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"[evaluate] Results saved to {path}")


def summarise_all(results: dict, binary: bool = True):
    """
    Print a compact leaderboard table sorted by recall (binary)
    or F1 (multiclass).
    """
    sort_key = "recall" if binary else "f1"
    ranked = sorted(results.items(),
                    key=lambda x: x[1].get(sort_key, 0),
                    reverse=True)

    print(f"\n{'='*70}")
    header = f"{'Model':<25} {'Acc':>7} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}"
    print(header)
    print("-" * 70)
    for name, m in ranked:
        auc_str = f"{m.get('auc_roc', 0):.4f}" if m.get("auc_roc") else "  N/A  "
        print(
            f"{name:<25} "
            f"{m.get('accuracy', 0):>7.4f} "
            f"{m.get('precision', 0):>10.4f} "
            f"{m.get('recall', 0):>8.4f} "
            f"{m.get('f1', 0):>8.4f} "
            f"{auc_str:>8}"
        )
    print("=" * 70)
    best = ranked[0][0]
    print(f"\n  Best model by {sort_key}: {best}\n")
