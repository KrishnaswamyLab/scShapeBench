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

from scshapebench.evaluator.loader import load_graphs
from scshapebench.evaluator.validator import validate_submission
from scshapebench.evaluator.metrics import evaluate, aggregate
from scshapebench.evaluator.io import save_results, store_result
from scshapebench.evaluator.baselines import baseline_predictions

from scshapebench.datasets.labels import load_labels
from scshapebench.features.persistence  import extract_features as extract_persistence_features
from scshapebench.features.graph_stats  import extract_features as extract_graph_features
from scshapebench.models import get_models


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

_CLASS_DISPLAY = {
    "clusters":     "CLUSTERS",
    "simple_traj":  "SINGLE_TRAJECTORY",
    "multi_branch": "MULTI_BRANCHING",
    "archetypal":   "ARCHETYPAL",
}

LABEL_PATHS = {
    "majority": "benchmarks/scrnaseq/labels/majority_labels.json",
    "confidence_score": "benchmarks/scrnaseq/labels/confidence_labels.json",
    "union": "benchmarks/scrnaseq/labels/union_labels.json",
    "high_support": "benchmarks/scrnaseq/labels/high_support_labels.json",
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


def make_folds(graphs, k=5, seed=42):
    ids = sorted(graphs.keys())

    rng = random.Random(seed)
    rng.shuffle(ids)

    folds = []

    for i in range(k):
        test_ids = set(ids[i::k]) 
        train_ids = [x for x in ids if x not in test_ids]

        folds.append({
            "train": train_ids,
            "test": list(test_ids)
        })

    return folds


def load_folds(splits_path: Path) -> list[dict[str, list[str]]]:
    data = json.loads(splits_path.read_text())
    return [
        {"train": data[key]["train"], "test": data[key]["test"]}
        for key in sorted(data.keys())
    ]


def _filter_split_ids(ids: list[str], available_ids: set[str], split_name: str) -> list[str]:
    kept = [sid for sid in ids if sid in available_ids]
    dropped = sorted(set(ids) - available_ids)
    if dropped:
        preview = ", ".join(dropped[:5])
        suffix = "..." if len(dropped) > 5 else ""
        print(f"  Dropped {len(dropped)} {split_name} ID(s) without both graph and label: {preview}{suffix}")
    return kept


def _set_model_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ──────────────────────────────────────────────────────────────────────────────
# Core runner
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    submission_dir: str,
    label_type: str,
    output_dir: str,
    device: str = "cpu",
    splits_path: str | None = "splits/folds_v1.json",
):
    submission_dir = Path(submission_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)

    print(f"Running scShapeBench on {submission_dir}")
    print(f"Labels ground truth using {label_type}")

    # 1. Load submission
    graphs, metadata = load_graphs(submission_dir)

    # 2. Validate submission
    validate_submission(graphs)

    # 3. Load labels + predefined splits
    label_path = LABEL_PATHS[label_type]

    labels, _, dataset_order = load_labels(path=label_path)
    id_to_idx = {sid: i for i, sid in enumerate(dataset_order)}
    evaluable_ids = set(graphs) & set(dataset_order)
    graph_only = sorted(set(graphs) - set(dataset_order))
    label_only = sorted(set(dataset_order) - set(graphs))
    if graph_only:
        print(f"Warning: excluding {len(graph_only)} graph(s) with no label: {', '.join(graph_only[:5])}")
    if label_only:
        print(f"Warning: excluding {len(label_only)} label(s) with no graph: {', '.join(label_only[:5])}")
    if not evaluable_ids:
        raise ValueError("No dataset IDs have both graphs and labels.")

    if splits_path and Path(splits_path).exists():
        print(f"Using predefined splits from {splits_path}")
        splits = load_folds(Path(splits_path))
    else:
        print("Using generated folds over evaluable graph/label IDs")
        splits = make_folds({sid: graphs[sid] for sid in evaluable_ids}, k=5, seed=42)
    all_results = {}

    # ── Cross-validation loop ────────────────────────────────────────────────    
    for fold_id, split in enumerate(splits):
        print(f"\n=== Fold {fold_id} ===")

        train_ids = _filter_split_ids(split["train"], evaluable_ids, "train")
        test_ids  = _filter_split_ids(split["test"], evaluable_ids, "test")

        train_graphs = [graphs[i] for i in train_ids]
        test_graphs  = [graphs[i] for i in test_ids]
        train_labels = labels[[id_to_idx[i] for i in train_ids]]
        test_labels  = labels[[id_to_idx[i] for i in test_ids]]

        if len(train_graphs) != len(train_labels) or len(test_graphs) != len(test_labels):
            raise RuntimeError("Internal ID alignment error between graphs and labels.")
        # 4. Feature extraction (each model declares which features it needs)
        features_cache = {
            "persistence": extract_persistence_features(train_graphs, test_graphs),
            "graph_stats": extract_graph_features(train_graphs, test_graphs),
        }

        for baseline_name, (preds, probs) in baseline_predictions(
            train_labels=train_labels,
            test_labels=test_labels,
            seed=20_000 + fold_id,
        ).items():
            store_result(
                all_results,
                model_name=baseline_name,
                fold_id=fold_id,
                metrics=evaluate(test_labels, preds, probs),
            )

        # 5. Evaluate all models
        for model_idx, model in enumerate(get_models(device)):
            _set_model_seed(10_000 + fold_id * 100 + model_idx)

            src = model.features_source
            X_train_m, X_test_m = features_cache[src] if src else (None, None)

            model.fit(
                graphs=train_graphs,
                labels=train_labels,
                features=X_train_m,
            )

            preds, probs = model.predict(
                graphs=test_graphs,
                features=X_test_m,
            )

            metrics = evaluate(test_labels, preds, probs)

            store_result(
                all_results,
                model_name=model.name,
                fold_id=fold_id,
                metrics=metrics,
            )

    # 6. Aggregate across folds
    final_results = aggregate(all_results)

    # 7. Save results
    save_results(final_results, output_dir, metadata)

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
