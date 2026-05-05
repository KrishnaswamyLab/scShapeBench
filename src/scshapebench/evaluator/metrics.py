import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scshapebench.datasets.labels import CLASS_ORDER

SUMMARY_KEY = "__summary__"


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
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(yt, yp))
        else:
            metrics["auc"] = float("nan")
            metrics["balanced_accuracy"] = float("nan")
        if yt.sum() > 0:
            metrics["auprc"] = float(average_precision_score(yt, ys))
        else:
            metrics["auprc"] = float("nan")

        results[cls_name] = metrics

    macro_metrics = {}
    for metric in ("accuracy", "balanced_accuracy", "f1", "auc", "auprc", "precision", "recall"):
        vals = [
            results[cls_name][metric]
            for cls_name in CLASS_ORDER
            if not np.isnan(results[cls_name].get(metric, float("nan")))
        ]
        macro_metrics[f"macro_{metric}"] = float(np.mean(vals)) if vals else float("nan")

    yt_all = y_true.astype(int)
    yp_all = y_pred.astype(int)
    macro_metrics.update({
        "micro_f1": float(f1_score(yt_all, yp_all, average="micro", zero_division=0)),
        "micro_precision": float(precision_score(yt_all, yp_all, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(yt_all, yp_all, average="micro", zero_division=0)),
        "exact_match": float(np.mean(np.all(yt_all == yp_all, axis=1))),
    })
    if yt_all.sum() > 0:
        macro_metrics["micro_auprc"] = float(average_precision_score(yt_all.ravel(), y_probs.ravel()))
    else:
        macro_metrics["micro_auprc"] = float("nan")
    if len(np.unique(yt_all.ravel())) > 1:
        macro_metrics["micro_auc"] = float(roc_auc_score(yt_all.ravel(), y_probs.ravel()))
    else:
        macro_metrics["micro_auc"] = float("nan")

    results[SUMMARY_KEY] = macro_metrics
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
            for metric in ("accuracy", "balanced_accuracy", "f1", "auc", "auprc", "precision", "recall"):
                vals = [
                    f[cls_name][metric]
                    for f in fold_list
                    if cls_name in f and not np.isnan(f[cls_name].get(metric, float("nan")))
                ]
                agg[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
                agg[f"{metric}_std"]  = float(np.std(vals))  if vals else float("nan")
            agg["folds"] = [f.get(cls_name, {}) for f in fold_list]
            per_class[cls_name] = agg
        summary = {}
        summary_keys = sorted({
            metric
            for f in fold_list
            for metric in f.get(SUMMARY_KEY, {}).keys()
        })
        for metric in summary_keys:
            vals = [
                f[SUMMARY_KEY][metric]
                for f in fold_list
                if SUMMARY_KEY in f and not np.isnan(f[SUMMARY_KEY].get(metric, float("nan")))
            ]
            summary[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            summary[f"{metric}_std"] = float(np.std(vals)) if vals else float("nan")
        summary["folds"] = [f.get(SUMMARY_KEY, {}) for f in fold_list]
        per_class[SUMMARY_KEY] = summary
        final[model_name] = per_class
    return final
