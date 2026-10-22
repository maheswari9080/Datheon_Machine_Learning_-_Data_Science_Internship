"""
Applied Machine Learning Studio - Publication-Grade Visualization Module
Provides customized, elegant Matplotlib/Seaborn visualizers adhering to the studio's editorial design tokens.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Editorial Design Tokens
COLOR_NAVY = "#0f172a"
COLOR_SLATE = "#334155"
COLOR_TEAL = "#0d9488"
COLOR_AMBER = "#d97706"
COLOR_CORAL = "#e11d48"
COLOR_GRID = "#e2e8f0"
COLOR_BG = "#ffffff"

plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": "#cbd5e1",
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "axes.labelsize": 11,
    "axes.labelcolor": COLOR_NAVY,
    "xtick.color": COLOR_SLATE,
    "ytick.color": COLOR_SLATE,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": COLOR_NAVY,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    classes: Sequence[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Render a polished heatmap visualization of a confusion matrix.
    """
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax,
        linewidths=1,
        linecolor="#f1f5f9",
    )
    ax.set_title(title, pad=15)
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Ground Truth", labelpad=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_decision_boundary_2d(
    model: Any,
    X_2d: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    title: str = "Decision Boundary (2D Projection)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 2D classification decision boundaries with data scatter overlay.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    if hasattr(Z[0], "isnumeric") or isinstance(Z[0], str):
        # Convert class strings to integer labels if needed
        label_map = {name: i for i, name in enumerate(class_names)}
        Z = np.array([label_map[item] for item in Z])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y if np.issubdtype(y.dtype, np.number) else [list(class_names).index(v) for v in y],
        cmap="viridis",
        edgecolor=COLOR_NAVY,
        linewidth=0.8,
        s=45,
    )

    ax.set_title(title, pad=15)
    ax.set_xlabel(feature_names[0], labelpad=10)
    ax.set_ylabel(feature_names[1], labelpad=10)

    # Custom legend
    handles = scatter.legend_elements()[0]
    ax.legend(handles, class_names, title="Classes", framealpha=0.9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs. Predicted Values",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot comparing ground-truth targets against model predictions with ideal parity line.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, color=COLOR_TEAL, edgecolors="none", s=25)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color=COLOR_CORAL, linestyle="--", linewidth=2, label="Ideal Fit (y = x)")

    ax.set_title(title, pad=15)
    ax.set_xlabel("True Observed Value", labelpad=10)
    ax.set_ylabel("Predicted Value", labelpad=10)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis & Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel regression diagnostic: Residual scatter against predictions and residual histogram.
    """
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.35, color=COLOR_TEAL, edgecolors="none", s=25)
    ax1.axhline(0, color=COLOR_AMBER, linestyle="--", linewidth=2)
    ax1.set_title("Residuals vs. Fitted Values")
    ax1.set_xlabel("Fitted (Predicted) Value")
    ax1.set_ylabel("Residual (True - Predicted)")

    # Distribution
    sns.histplot(residuals, kde=True, color=COLOR_NAVY, ax=ax2, bins=35)
    ax2.set_title("Residual Error Distribution")
    ax2.set_xlabel("Residual Error")
    ax2.set_ylabel("Density")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Classifier",
    title: str = "Receiver Operating Characteristic (ROC)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curve with calculated Area Under Curve (AUC).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=COLOR_TEAL, linewidth=2.5, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=COLOR_SLATE, linestyle="--", linewidth=1.5, label="Random Chance (AUC = 0.500)")

    ax.set_title(title, pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", labelpad=10)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", labelpad=10)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_feature_importance(
    feature_names: Sequence[str],
    importances: Sequence[float],
    top_n: int = 10,
    title: str = "Top Feature Importances",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart depicting top predictive features sorted by magnitude.
    """
    df_imp = pd.DataFrame({"Feature": list(feature_names), "Importance": list(importances)})
    df_imp = df_imp.sort_values(by="Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_imp, y="Feature", x="Importance", color=COLOR_TEAL, ax=ax)
    ax.set_title(title, pad=15)
    ax.set_xlabel("Importance Metric", labelpad=10)
    ax.set_ylabel("Feature Name", labelpad=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig
