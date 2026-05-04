"""
Benchmark runner for scShapeBench.

Evaluates a user submission (graphs only) using fixed splits and
standardized downstream models.

Usage
-----
python run_benchmark.py --submission submissions/scRNAseq/scReebTower --output results/
"""

import argparse
import math
from pathlib import Path

import torch

import random

from scshapebench.evaluator.loader import load_graphs
from scshapebench.evaluator.validator import validate_submission
from scshapebench.evaluator.metrics import evaluate, aggregate
from scshapebench.evaluator.io import save_results, store_result

from scshapebench.datasets.labels import load_labels
from scshapebench.features.persistence import extract_features
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


def _fmt(val: float) -> str:
    return f"{val:.4f}" if not math.isnan(val) else "  nan "


def _print_results(final_results: dict, metadata: dict) -> None:
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
        acc  = r.get("accuracy_mean",  float("nan"))
        f1   = r.get("f1_mean",        float("nan"))
        auc  = r.get("auc_mean",       float("nan"))
        prec = r.get("precision_mean", float("nan"))
        rec  = r.get("recall_mean",    float("nan"))
        print(f"  {display:<24}{acc:9.4f}{f1:9.4f}{auc:9.4f}{prec:9.4f}{rec:9.4f}")

    # ── Benchmark results (all models) ──────────────────────────────────
    print("\n" + "=" * 58)
    print("  BENCHMARK RESULTS  (mean over CV folds)")
    print("=" * 58)
    print(f"  {'Model':<5}  {'Method':<8}  {'Class':<24}  {'F1':>7}  {'AUC':>7}  {'Acc':>7}")
    print("  " + "-" * 56)

    first = True
    for model_name in ("mlp", "svm", "gnn"):
        if model_name not in final_results:
            continue
        if not first:
            print()
        first = False
        for cls_name, display in _CLASS_DISPLAY.items():
            if cls_name not in final_results[model_name]:
                continue
            r = final_results[model_name][cls_name]
            f1  = r.get("f1_mean",       float("nan"))
            auc = r.get("auc_mean",      float("nan"))
            acc = r.get("accuracy_mean", float("nan"))
            print(f"  {model_name:<5}  {method:<8}  {display:<24}  {f1:7.4f}  {auc:7.4f}  {acc:7.4f}")

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

# ──────────────────────────────────────────────────────────────────────────────
# Core runner
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(submission_dir: str, output_dir: str, device: str = "cpu"):
    submission_dir = Path(submission_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)

    # 1. Load submission
    graphs, metadata = load_graphs(submission_dir)

    # 2. Validate submission
    validate_submission(graphs)

    # 3. Load labels + predefined splits
    labels, _, dataset_order = load_labels()
    id_to_idx = {sid: i for i, sid in enumerate(dataset_order)}
    # splits = load_splits("splits/folds_v1.json")
    splits = make_folds(graphs, k=5, seed=42)
    all_results = {}

    # ── Cross-validation loop ────────────────────────────────────────────────    
    for fold_id, split in enumerate(splits):
        print(f"\n=== Fold {fold_id} ===")

        train_ids = split["train"]
        test_ids  = split["test"]


        missing_train = set(train_ids) - set(graphs)
        missing_test = set(test_ids) - set(graphs)

        if missing_train or missing_test:
            raise ValueError(
                f"Split mismatch!\n"
                f"Missing train: {len(missing_train)}\n"
                f"Missing test: {len(missing_test)}"
            )

        train_graphs = [graphs[i] for i in train_ids if i in graphs]
        test_graphs  = [graphs[i] for i in test_ids  if i in graphs]

        train_labels = labels[[id_to_idx[i] for i in train_ids]]
        test_labels  = labels[[id_to_idx[i] for i in test_ids]]

        # 4. Feature extraction (for MLP/SVM)
        X_train, X_test = extract_features(train_graphs, test_graphs)

        # 5. Evaluate all models
        for model in get_models(device):
            print(f"  → Model: {model.name}")

            model.fit(
                graphs=train_graphs,
                labels=train_labels,
                features=X_train,
            )

            preds, probs = model.predict(
                graphs=test_graphs,
                features=X_test,
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

    _print_results(final_results,metadata)
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
        "--output",
        default="results/",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda",
    )

    args = parser.parse_args()

    run_benchmark(
        submission_dir=args.submission,
        output_dir=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()