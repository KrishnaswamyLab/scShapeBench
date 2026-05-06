"""
Benchmark runner for scShapeBench.

Evaluates a user submission (graphs only) using fixed splits and
standardized downstream models.

Usage
-----
python run_benchmark_single_cell.py --submission submissions/scRNAseq/scReebTower --label-type confidence_score --output results/
"""

import argparse
import math
import json
from pathlib import Path

import numpy as np
import torch

import random

import os

from scshapebench.evaluator.runner import run_real_submission


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

_CLASS_DISPLAY = {
    "clusters":     "CLUSTERS",
    "simple_traj":  "SINGLE_TRAJECTORY",
    "multi_branch": "MULTI_BRANCHING",
    "archetypal":   "ARCHETYPAL",
}


def _fmt_pm(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "   nan   "
    if math.isnan(std):
        return f"{mean:.4f} ±  nan"
    return f"{mean:.4f}±{std:.4f}"

def _print_results(final_results: dict, metadata: dict,label_type) -> None:
    method = metadata.get("method", "unknown")

    # ── Cross-validation summary (GNN) ──────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Cross-validation summary  [GNN  method={method}]")
    print("=" * 60)
    print(f"  {'Class':<24}{'Acc':>9}{'F1':>9}{'AUC':>9}{'Prec':>9}{'Rec':>9}")
    print("  " + "-" * 58)

    gnn = final_results.get("gnn", {})
    for cls_name, display in _CLASS_DISPLAY.items():
        if cls_name not in gnn:
            continue
        r = gnn[cls_name]

        acc  = _fmt_pm(r.get("accuracy_mean", float("nan")),
                    r.get("accuracy_std",  float("nan")))
        f1   = _fmt_pm(r.get("f1_mean", float("nan")),
                    r.get("f1_std",  float("nan")))
        auc  = _fmt_pm(r.get("auc_mean", float("nan")),
                    r.get("auc_std",  float("nan")))
        prec = _fmt_pm(r.get("precision_mean", float("nan")),
                    r.get("precision_std",  float("nan")))
        rec  = _fmt_pm(r.get("recall_mean", float("nan")),
                    r.get("recall_std",  float("nan")))

        print(f"  {display:<24}{acc:>15}{f1:>15}{auc:>15}{prec:>15}{rec:>15}")

    # ── Benchmark results (all models) ──────────────────────────────────
    print("\n" + "=" * 58)
    print(f"  BENCHMARK RESULTS {label_type} (mean over CV folds)")
    print("=" * 58)
    print(f"  {'Model':<25}  {'Method':<8}  {'Class':<24}  {'Acc':>15} {'BalAcc':>15} {'F1':>15}  {'AUPRC':>15} {'AUC':>15}")
    print("  " + "-" * 80)

    first = True
    for model_name in (
        "baseline_all_zero",
        "baseline_all_one",
        "baseline_train_prevalence",
        "baseline_random_prevalence",
        "gnn",
        "mlp",
        "pi_svm",
        "stats_svm",
        "rf",
    ):
        if model_name not in final_results:
            continue
        if not first:
            print()
        first = False
        for cls_name, display in _CLASS_DISPLAY.items():
            if cls_name not in final_results[model_name]:
                continue
            r = final_results[model_name][cls_name]
            f1  = _fmt_pm(r.get("f1_mean", float("nan")),
                        r.get("f1_std",  float("nan")))
            auc = _fmt_pm(r.get("auc_mean", float("nan")),
                        r.get("auc_std",  float("nan")))
            auprc = _fmt_pm(r.get("auprc_mean", float("nan")),
                        r.get("auprc_std",  float("nan")))
            acc = _fmt_pm(r.get("accuracy_mean", float("nan")),
                        r.get("accuracy_std",  float("nan")))
            bal_acc = _fmt_pm(r.get("balanced_accuracy_mean", float("nan")),
                        r.get("balanced_accuracy_std",  float("nan")))

            print(f"  {model_name:<25}  {method:<8}  {display:<24} {acc:>15} {bal_acc:>15} {f1:>15}  {auprc:>15} {auc:>15}")

    print("=" * 58)


def run_benchmark(
    submission_dir: str,
    label_type: str,
    output_dir: str,
    device: str = "cpu",
    splits_path: str | None = "splits/folds_v1.json",
):
    final_results, metadata = run_real_submission(
        submission_dir,
        label_type=label_type,
        output_dir=output_dir,
        device=device,
        splits_path=Path(splits_path) if splits_path else None,
    )

    _print_results(final_results,metadata,label_type)
    print(f"\nResults saved to: {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the scShapeBench benchmark on a submission."
    )
    parser.add_argument(
        "--submission",
        required=True,
        help="Path to submission directory",
    )
    parser.add_argument(
        "--label-type",
        choices=["majority", "confidence_score", "union", "high_support"],
        default="majority",
        help="Which label set to use"
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--splits",
        default="splits/folds_v1.json",
        help="Path to predefined folds JSON. Use an empty string to generate folds.",
    )

    args = parser.parse_args()

    run_benchmark(
        submission_dir=args.submission,
        output_dir=args.output,
        label_type=args.label_type,
        device=args.device,
        splits_path=args.splits or None,
    )


if __name__ == "__main__":
    main()
