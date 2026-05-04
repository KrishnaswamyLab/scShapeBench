"""
Benchmark runner for scShapeBench.

Iterates over all combinations of (model × graph-method × k) and reports
binary classification metrics (Accuracy / F1 / AUC / Precision / Recall)
for each TopologyClass independently.

Usage
-----
    # Run everything with defaults
    python benchmark.py

    # Run specific combinations
    python benchmark.py --cv-folds 10 --output-dir results/

    # More CV folds, custom output
    python benchmark.py --cv-folds 10 --output-dir results/
"""

import argparse
import json
import pathlib
import sys
from datetime import datetime

import numpy as np

from dataset import (
    CLASS_ORDER,
    VALID_K,
    VALID_METHODS,
    build_feature_matrix,
    discover_samples,
)
import torch

MODELS = ("mlp", "svm", "gnn")

_FEATURE_MODELS = {"mlp", "svm"}   # take (X, Y) numpy arrays
_GRAPH_MODELS   = {"gnn"}          # take raw samples list


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_model(name: str):
    if name == "mlp":
        from models import persistence_mlp
        return persistence_mlp
    if name == "svm":
        from models import persistence_svm
        return persistence_svm
    if name == "gnn":
        from models import gnn
        return gnn
    raise ValueError(f"Unknown model '{name}'. Choices: {MODELS}")


# ── Results aggregation ───────────────────────────────────────────────────────

def _aggregate(fold_results: dict[str, list[dict]]) -> dict[str, dict]:
    """Compute mean ± std for each metric over folds, per class."""
    out = {}
    for cls, folds in fold_results.items():
        agg = {}
        for metric in ("accuracy", "f1", "auc", "precision", "recall"):
            vals = [f[metric] for f in folds if not np.isnan(f.get(metric, float("nan")))]
            agg[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            agg[f"{metric}_std"]  = float(np.std(vals))  if vals else float("nan")
        agg["folds"] = folds
        out[cls] = agg
    return out


# ── Table printer ─────────────────────────────────────────────────────────────

def _print_summary_table(rows: list[dict]) -> None:
    """
    rows: list of {model, method, k, cls, agg}
    Prints a compact table sorted by (model, method, k, cls).
    """
    col_w = 58
    header = (
        f"  {'Model':<6} {'Method':<8} {'K':<12} {'Class':<22}"
        f"  {'F1':>7}  {'AUC':>7}  {'Acc':>7}"
    )
    print(f"\n{'='*col_w}")
    print("  BENCHMARK RESULTS  (mean over CV folds)")
    print(f"{'='*col_w}")
    print(header)
    print(f"  {'-'*(col_w-2)}")

    cur_group = None
    for r in rows:
        group = (r["model"], r["method"], r["k"])
        if group != cur_group:
            if cur_group is not None:
                print()
            cur_group = group
        a = r["agg"]
        f1_s  = f"{a['f1_mean']:7.4f}" if not np.isnan(a['f1_mean'])  else "    N/A"
        auc_s = f"{a['auc_mean']:7.4f}" if not np.isnan(a['auc_mean']) else "    N/A"
        acc_s = f"{a['accuracy_mean']:7.4f}"
        print(
            f"  {r['model']:<6} {r['method']:<8} {r['k']:<12} {r['cls']:<22}"
            f"  {f1_s}  {auc_s}  {acc_s}"
        )
    print(f"{'='*col_w}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the scShapeBench benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",    default="data",  help="Root data directory")
    parser.add_argument("--labels-json", default=None,    help="Labels JSON path (auto-detected if omitted)")
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS),
        metavar="MODEL", help=f"Models to run. Choices: {MODELS}",
    )
    parser.add_argument(
        "--graph-methods", nargs="+", default=["dreeb"], choices=list(VALID_METHODS),
        metavar="METHOD", help=f"Graph methods. Choices: {VALID_METHODS}",
    )
    parser.add_argument(
        "--k", nargs="+", default=["k25"], choices=list(VALID_K),
        metavar="K", help=f"k-NN settings. Choices: {VALID_K}",
    )
    parser.add_argument("--cv-folds",   type=int, default=5,         help="Number of CV folds")
    parser.add_argument("--output-dir", default="results",           help="Directory for JSON results")
    args = parser.parse_args()

    root     = pathlib.Path(__file__).parent
    data_dir = str(root / args.data_dir)

    if args.labels_json:
        labels_json = args.labels_json
    else:
        candidates = sorted(root.glob("phate_gallery_labels*/labels.json"))
        if not candidates:
            print("ERROR: labels JSON not found. Use --labels-json.", file=sys.stderr)
            sys.exit(1)
        labels_json = str(candidates[-1])
        print(f"Labels: {labels_json}\n")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    all_results: dict = {}
    table_rows:  list = []

    for k in args.k:
        for method in args.graph_methods:
            print(f"\n{'#'*60}")
            print(f"#  k={k}  method={method}")
            print(f"{'#'*60}")

            samples = discover_samples(data_dir, labels_json, k=k, method=method)
            if not samples:
                print("  No samples found — skipping.")
                continue

            X, Y = None, None
            feature_models = [m for m in args.models if m in _FEATURE_MODELS]
            if feature_models:
                print("\nBuilding feature matrix...")
                X, Y = build_feature_matrix(samples, k=k, method=method)
                print(f"X: {X.shape}  Y: {Y.shape}\n")

            for model_name in args.models:
                run_key = f"{model_name}|{method}|{k}"
                print(f"\n{'─'*50}")
                print(f"  Model: {model_name.upper()}  method: {method}  k: {k}")
                print(f"{'─'*50}")

                mod = _load_model(model_name)

                if model_name in _GRAPH_MODELS:
                    fold_results = mod.run(
                        samples, k=k, method=method,
                        k_folds=args.cv_folds, device=device,
                    )
                elif model_name == "mlp":
                    fold_results = mod.run(X, Y, k_folds=args.cv_folds, device=device)
                else:
                    fold_results = mod.run(X, Y, k_folds=args.cv_folds)

                agg = _aggregate(fold_results)
                all_results[run_key] = {
                    "model": model_name, "method": method, "k": k,
                    "per_class": agg,
                }
                for cls in CLASS_ORDER:
                    table_rows.append({
                        "model": model_name, "method": method, "k": k,
                        "cls": cls, "agg": agg[cls],
                    })

    _print_summary_table(table_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = output_dir / f"benchmark_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
