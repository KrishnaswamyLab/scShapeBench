import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from scshapebench.datasets.labels import CLASS_ORDER


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> dict:
    """
    Compute per-class binary metrics.

    Args:
        y_true:  [N, C] ground-truth binary labels
        y_pred:  [N, C] binary predictions
        y_probs: [N, C] probability scores

    Returns:
        {class_name: {metric: value}}
    """
    results = {}
    for i, cls_name in enumerate(CLASS_ORDER):
        yt = y_true[:, i].astype(int)
        yp = y_pred[:, i].astype(int)
        ys = y_probs[:, i]

        metrics = {
            "accuracy":  float(accuracy_score(yt, yp)),
            "f1":        float(f1_score(yt, yp, zero_division=0)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall":    float(recall_score(yt, yp, zero_division=0)),
        }
        if len(np.unique(yt)) > 1:
            metrics["auc"] = float(roc_auc_score(yt, ys))
        else:
            metrics["auc"] = float("nan")

        results[cls_name] = metrics
    return results


def aggregate(all_results: dict) -> dict:
    """
    Aggregate per-fold metrics across folds, per model and per class.

    Args:
        all_results: {model_name: {fold_id: {class_name: {metric: value}}}}

    Returns:
        {model_name: {class_name: {metric_mean, metric_std, folds}}}
    """
    final = {}
    for model_name, folds in all_results.items():
        fold_list = list(folds.values())
        per_class = {}
        for cls_name in CLASS_ORDER:
            agg = {}
            for metric in ("accuracy", "f1", "auc", "precision", "recall"):
                vals = [
                    f[cls_name][metric]
                    for f in fold_list
                    if cls_name in f and not np.isnan(f[cls_name].get(metric, float("nan")))
                ]
                agg[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
                agg[f"{metric}_std"]  = float(np.std(vals))  if vals else float("nan")
            agg["folds"] = [f.get(cls_name, {}) for f in fold_list]
            per_class[cls_name] = agg
        final[model_name] = per_class
    return final
