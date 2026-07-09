"""
src/visualise.py
All plotting functions for the IDS project.
Saves figures to config.PLOT_DIR.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config

STYLE = {
    "fig_bg":   "#0d1117",
    "ax_bg":    "#161b22",
    "grid":     "#21262d",
    "text":     "#c9d1d9",
    "border":   "#30363d",
    "palette":  ["#3b82f6", "#10b981", "#f59e0b", "#ef4444",
                 "#8b5cf6", "#06b6d4", "#f97316", "#ec4899"],
}

def _apply_dark(fig, axes):
    fig.patch.set_facecolor(STYLE["fig_bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(STYLE["ax_bg"])
        ax.tick_params(colors=STYLE["text"])
        ax.xaxis.label.set_color(STYLE["text"])
        ax.yaxis.label.set_color(STYLE["text"])
        ax.title.set_color(STYLE["text"])
        for spine in ax.spines.values():
            spine.set_color(STYLE["border"])
        ax.grid(color=STYLE["grid"], linewidth=0.5, linestyle="--")


def plot_class_distribution(y: np.ndarray, label_names: list = None,
                              title: str = "Class Distribution",
                              filename: str = "class_distribution.png"):
    unique, counts = np.unique(y, return_counts=True)
    labels = [label_names[i] if label_names else str(i) for i in unique]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts,
                  color=STYLE["palette"][:len(unique)],
                  edgecolor="none", width=0.6)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count:,}", ha="center", va="bottom",
                color=STYLE["text"], fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Sample count")
    plt.xticks(rotation=30, ha="right")
    _apply_dark(fig, ax)
    plt.tight_layout()
    _save(fig, filename)


def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                           model_name: str = "",
                           filename: str = "confusion_matrix.png",
                           normalise: bool = True):
    if normalise:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2),
                                     max(5, len(class_names))))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, ax=ax,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor=STYLE["border"],
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    _apply_dark(fig, ax)
    plt.tight_layout()
    _save(fig, filename)


def plot_model_comparison(results: dict,
                           metrics: list = None,
                           filename: str = "model_comparison.png"):
    metrics = metrics or ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [results[m].get(metric, 0) for m in model_names]
        colors = STYLE["palette"][:len(model_names)]
        bars = ax.barh(model_names, values, color=colors,
                       edgecolor="none", height=0.5)
        ax.set_xlim(0, 1.1)
        ax.set_title(metric.capitalize())
        for bar, val in zip(bars, values):
            ax.text(val + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center",
                    color=STYLE["text"], fontsize=9)

    fig.suptitle("Model Comparison", color=STYLE["text"],
                 fontsize=14, fontweight="bold")
    _apply_dark(fig, axes)
    plt.tight_layout()
    _save(fig, filename)


def plot_feature_importance(importances: list, top_n: int = 20,
                             model_name: str = "Random Forest",
                             filename: str = "feature_importance.png"):
    """importances: list of (feature_name, score) sorted descending."""
    top = importances[:top_n]
    names  = [x[0] for x in top]
    scores = [x[1] for x in top]

    cmap   = plt.cm.plasma(np.linspace(0.3, 0.9, len(names)))
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(names[::-1], scores[::-1], color=cmap, edgecolor="none")
    for bar, val in zip(bars, scores[::-1]):
        ax.text(val + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center",
                color=STYLE["text"], fontsize=8)
    ax.set_xlabel("Importance score")
    ax.set_title(f"Top {top_n} Features — {model_name}")
    _apply_dark(fig, ax)
    plt.tight_layout()
    _save(fig, filename)


def plot_roc_curves(roc_data: dict, filename: str = "roc_curves.png"):
    """
    roc_data: {model_name: (fpr, tpr, auc_score)}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color=STYLE["border"], linewidth=1)

    for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        color = STYLE["palette"][i % len(STYLE["palette"])]
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=9,
              facecolor=STYLE["ax_bg"], labelcolor=STYLE["text"])
    _apply_dark(fig, ax)
    plt.tight_layout()
    _save(fig, filename)


def plot_training_history(train_losses: list, val_losses: list = None,
                           filename: str = "nn_training_history.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=STYLE["palette"][0],
            linewidth=2, label="Train loss")
    if val_losses:
        ax.plot(epochs, val_losses, color=STYLE["palette"][1],
                linewidth=2, linestyle="--", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Neural Network Training History")
    ax.legend(facecolor=STYLE["ax_bg"], labelcolor=STYLE["text"])
    _apply_dark(fig, ax)
    plt.tight_layout()
    _save(fig, filename)


def plot_precision_recall_tradeoff(precisions: list, recalls: list,
                                    thresholds: list,
                                    model_name: str = "",
                                    filename: str = "pr_tradeoff.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, precisions[:-1], color=STYLE["palette"][0],
            linewidth=2, label="Precision")
    ax.plot(thresholds, recalls[:-1], color=STYLE["palette"][3],
            linewidth=2, label="Recall")
    ax.axvline(x=0.5, color=STYLE["border"], linestyle="--",
               linewidth=1, label="Default threshold (0.5)")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision-Recall Tradeoff — {model_name}")
    ax.legend(facecolor=STYLE["ax_bg"], labelcolor=STYLE["text"])
    _apply_dark(fig, ax)
    plt.tight_layout()
    _save(fig, filename)


def _save(fig, filename: str):
    path = os.path.join(config.PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualise] Saved {path}")
