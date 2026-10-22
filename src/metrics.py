"""
Applied Machine Learning Studio - Model Evaluation & Metrics Engine
Provides standardized metric calculations, JSON export, and tabular reporting for classification and regression tasks.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, pd.Series, List[Any]],
    y_pred: Union[np.ndarray, pd.Series, List[Any]],
    y_prob: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    pos_label: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for binary or multi-class classification.

    Parameters
    ----------
    y_true : array-like
        True ground-truth labels.
    y_pred : array-like
        Predicted class labels.
    y_prob : Optional[array-like]
        Predicted class probabilities.
    pos_label : Optional[Any]
        Positive class indicator for binary metrics.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed performance metrics.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    classes = np.unique(y_true_arr)
    is_binary = len(classes) == 2

    metrics: Dict[str, Any] = {
        "accuracy": float(round(accuracy_score(y_true_arr, y_pred_arr), 4)),
        "precision_macro": float(round(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0), 4)),
        "recall_macro": float(round(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0), 4)),
        "f1_macro": float(round(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0), 4)),
        "precision_weighted": float(round(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0), 4)),
        "recall_weighted": float(round(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0), 4)),
        "f1_weighted": float(round(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0), 4)),
    }

    if is_binary and pos_label is not None:
        metrics["precision_binary"] = float(
            round(precision_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0), 4)
        )
        metrics["recall_binary"] = float(
            round(recall_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0), 4)
        )
        metrics["f1_binary"] = float(
            round(f1_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0), 4)
        )

    # ROC-AUC computation
    metrics["roc_auc"] = None
    if y_prob is not None:
        try:
            if is_binary:
                # Expect 1D array of positive class probability or 2D array
                if hasattr(y_prob, "ndim") and y_prob.ndim == 2:
                    prob_pos = y_prob[:, 1]
                else:
                    prob_pos = y_prob
                metrics["roc_auc"] = float(round(roc_auc_score(y_true_arr, prob_pos), 4))
            else:
                metrics["roc_auc"] = float(
                    round(roc_auc_score(y_true_arr, y_prob, multi_class="ovr", average="macro"), 4)
                )
        except Exception:
            metrics["roc_auc"] = None

    cm = confusion_matrix(y_true_arr, y_pred_arr)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classes"] = [str(c) for c in classes]
    metrics["report"] = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)

    return metrics


def compute_regression_metrics(
    y_true: Union[np.ndarray, pd.Series, List[float]],
    y_pred: Union[np.ndarray, pd.Series, List[float]],
    n_features: Optional[int] = None,
) -> Dict[str, float]:
    """
    Calculate standard regression evaluation metrics including RMSE, MAE, R², and adjusted R².
    """
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    n = len(y_true_arr)

    mse = mean_squared_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    r2 = r2_score(y_true_arr, y_pred_arr)

    metrics = {
        "rmse": float(round(rmse, 4)),
        "mse": float(round(mse, 4)),
        "mae": float(round(mae, 4)),
        "r2": float(round(r2, 4)),
    }

    if n_features is not None and n > n_features + 1:
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1)
        metrics["adjusted_r2"] = float(round(adj_r2, 4))

    return metrics


def export_metrics_json(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Save evaluation metrics dictionary to formatted JSON file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def metrics_summary_table(models_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Format multiple model metric dictionaries into a comparative Pandas DataFrame.
    """
    records = []
    for model_name, metrics in models_metrics.items():
        row = {"Model": model_name}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                row[k.upper()] = v
        records.append(row)
    return pd.DataFrame(records).set_index("Model")
