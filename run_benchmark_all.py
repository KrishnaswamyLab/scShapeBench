"""
Run scShapeBench across all methods under submissions/scRNAseq/ and print a
combined comparison table.

Usage
-----
python run_benchmark_all.py --label-type confidence_score --splits splits/folds_v1.json --output results/
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import random

from scshapebench.evaluator.loader import load_graphs
from scshapebench.evaluator.validator import validate_submission
from scshapebench.evaluator.metrics import evaluate, aggregate
from scshapebench.evaluator.io import save_results, store_result
from scshapebench.evaluator.baselines import baseline_predictions
from scshapebench.evaluator.runner import (
    LABEL_PATHS,
    run_real_collection,
    run_real_submission,
)
from scshapebench.datasets.labels import load_labels
from scshapebench.features.persistence import extract_features as extract_persistence_features
from scshapebench.features.graph_stats import extract_features as extract_graph_features
from scshapebench.models import get_models


_CLASS_DISPLAY = {
    "clusters":     "CLUSTERS",
    "simple_traj":  "SINGLE_TRAJECTORY",
    "multi_branch": "MULTI_BRANCHING",
    "archetypal":   "ARCHETYPAL",
}

MODEL_ORDER = ("gnn", "mlp", "stats_svm")
CONSOLE_MODEL_ORDER = (
    "baseline_all_zero",
    "baseline_all_one",
    "baseline_train_prevalence",
    "baseline_random_prevalence",
    "gnn",
    "mlp",
    "pi_svm",
    "stats_svm",
    "rf",
)

# column order within each model group in the LaTeX table
_CLASS_LATEX_ORDER = [
    ("clusters",     "Cl"),
    ("simple_traj",  "ST"),
    ("multi_branch", "MT"),
    ("archetypal",   "A"),
]

_MODEL_DISPLAY_NAMES = {
    "mlp":       "MLP",
    "pi_svm":    "PI-SVM",
    "stats_svm": "Stats-SVM",
    "gnn":       "GNN",
    "rf":        "RF",
}


def _no_lead_zero(v: float) -> str:
    s = f"{v:.2f}"
    return s[1:] if s.startswith("0.") else s


def _fmt_latex_val(mean: float, std: float) -> str:
    if math.isnan(mean) or math.isnan(std):
        return r"\text{---}"
    return f"{_no_lead_zero(mean)}$\\pm${_no_lead_zero(std)}"


def _fmt_pm(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "   nan   "
    if math.isnan(std):
        return f"{mean:.4f} ±  nan"
    return f"{mean:.4f}±{std:.4f}"


def make_folds(graphs, k=5, seed=42):
    ids = sorted(graphs.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    folds = []
    for i in range(k):
        test_ids = set(ids[i::k])
        train_ids = [x for x in ids if x not in test_ids]
        folds.append({"train": train_ids, "test": list(test_ids)})
    return folds


def load_folds(splits_path: Path) -> list:
    data = json.loads(splits_path.read_text())
    folds = []
    for key in sorted(data.keys()):
        folds.append({"train": data[key]["train"], "test": data[key]["test"]})
    return folds


def _filter_split_ids(ids: list[str], available_ids: set[str], split_name: str) -> list[str]:
    kept = [sid for sid in ids if sid in available_ids]
    dropped = sorted(set(ids) - available_ids)
    if dropped:
        preview = ", ".join(dropped[:5])
        suffix = "..." if len(dropped) > 5 else ""
        print(f"    Dropped {len(dropped)} {split_name} ID(s) without both graph and label: {preview}{suffix}")
    return kept


def _set_model_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single(submission_dir: Path, label_type: str, output_dir: Path, device, splits_path: Path = None):
    return run_real_submission(
        submission_dir,
        label_type=label_type,
        output_dir=output_dir,
        device=device,
        splits_path=splits_path,
    )


def _print_comparison_table(all_method_results: list, label_type: str) -> None:
    sep = "=" * 68
    print("\n" + sep)
    print(f"  BENCHMARK RESULTS {label_type} (mean over CV folds)")
    print(sep)
    print(f"  {'Model':<27}  {'Method':<14}  {'Class':<24}  {'Acc':>15} {'BalAcc':>15} {'F1':>15}  {'AUPRC':>15} {'AUC':>15}")
    print("  " + "-" * 84)

    for final_results, metadata in all_method_results:
        method = metadata.get("method", "unknown")
        first = True
        for model_name in CONSOLE_MODEL_ORDER:
            if model_name not in final_results:
                continue
            if not first:
                print()
            first = False
            for cls_name, display in _CLASS_DISPLAY.items():
                if cls_name not in final_results[model_name]:
                    continue
                r = final_results[model_name][cls_name]
                acc = _fmt_pm(r.get("accuracy_mean", float("nan")), r.get("accuracy_std", float("nan")))
                bal_acc = _fmt_pm(r.get("balanced_accuracy_mean", float("nan")), r.get("balanced_accuracy_std", float("nan")))
                f1 = _fmt_pm(r.get("f1_mean", float("nan")), r.get("f1_std", float("nan")))
                auprc = _fmt_pm(r.get("auprc_mean", float("nan")), r.get("auprc_std", float("nan")))
                auc = _fmt_pm(r.get("auc_mean", float("nan")), r.get("auc_std", float("nan")))
                print(f"  {model_name:<27}  {method:<14}  {display:<24} {acc:>15} {bal_acc:>15} {f1:>15}  {auprc:>15} {auc:>15}")

        if (final_results, metadata) != all_method_results[-1]:
            print()

    print(sep)


def _print_latex_table(
    all_method_results: list,
    label_type: str,
    out_path: Path,
    our_method: str = None,
) -> None:
    models_present = [m for m in MODEL_ORDER if any(m in r for r, _ in all_method_results)]
    n = len(models_present)

    # rank on F1 per (model, class) to find best / second-best
    col_scores: dict = {}
    for final_results, metadata in all_method_results:
        method = metadata.get("method", "unknown")
        for mdl in models_present:
            for cls_name, _ in _CLASS_LATEX_ORDER:
                for metric in ("accuracy", "f1"):
                    mean = final_results.get(mdl, {}).get(cls_name, {}).get(f"{metric}_mean", float("nan"))
                    col_scores.setdefault((mdl, cls_name, metric), []).append((method, mean))

    best, second = {}, {}
    for key, entries in col_scores.items():
        ranked = sorted([(m, v) for m, v in entries if not math.isnan(v)], key=lambda x: x[1], reverse=True)
        if ranked:
            best[key] = ranked[0][0]
        if len(ranked) >= 2:
            second[key] = ranked[1][0]

    def _decorate(mdl, cls_name, metric, method, mean, std):
        cell = _fmt_latex_val(mean, std)
        key = (mdl, cls_name, metric)
        if best.get(key) == method:
            return r"\textbf{" + cell + "}"
        if second.get(key) == method:
            return r"\underline{" + cell + "}"
        return cell

    # 2 metrics per class → 8 cols per model group, plus a separator between groups
    # col spec:  l  cccccccc c  cccccccc c  cccccccc
    col_spec = "l " + " ".join("cccccccc" + (" c" if i < n - 1 else "") for i in range(n))

    # each model group spans 8 cols; col index of group i starts at 2 + i*9
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + i * 9}-{9 + i * 9}}}" for i in range(n)
    )

    model_header_cells = []
    for i, mdl in enumerate(models_present):
        model_header_cells.append(f"\\multicolumn{{8}}{{c}}{{{_MODEL_DISPLAY_NAMES.get(mdl, mdl)}}}")
        if i < n - 1:
            model_header_cells.append("")
    model_header = "& " + " & ".join(model_header_cells) + r" \\"

    # second header row: class labels spanning 2 cols each
    class_header_cells = [""]
    for i, _ in enumerate(models_present):
        for _, col_label in _CLASS_LATEX_ORDER:
            class_header_cells.append(f"\\multicolumn{{2}}{{c}}{{{col_label}}}")
        if i < n - 1:
            class_header_cells.append("")
    class_header = " & ".join(class_header_cells) + r" \\"

    # cmidrules under each class label (pairs of cols)
    class_cmidrules_parts = []
    for i in range(n):
        group_start = 2 + i * 9
        for j in range(len(_CLASS_LATEX_ORDER)):
            s = group_start + j * 2
            class_cmidrules_parts.append(f"\\cmidrule(lr){{{s}-{s + 1}}}")
    class_cmidrules = " ".join(class_cmidrules_parts)

    # third header row: Acc / F1 repeated
    metric_header_cells = ["Method"]
    for i, _ in enumerate(models_present):
        for _ in _CLASS_LATEX_ORDER:
            metric_header_cells.extend(["Acc", "F1"])
        if i < n - 1:
            metric_header_cells.append("")
    metric_header = " & ".join(metric_header_cells) + r" \\"

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Per-shape accuracy and F1 on the scRNAseq track of scShapeBench "
        rf"(\textbf{{label type:}} {label_type}). "
        rf"Shape classes: \textbf{{A}}rchetypal, \textbf{{Cl}}usters, "
        rf"\textbf{{S}}imple \textbf{{T}}rajectory, \textbf{{M}}ultiple \textbf{{T}}rajectories. "
        rf"Best per column in \textbf{{bold}}; second best \underline{{underlined}}. "
        rf"Values are mean $\pm$ std over 5 CV folds.}}"
    )
    lines.append(r"\label{tab:scrnaseq-results}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(model_header)
    lines.append(cmidrules)
    lines.append(class_header)
    lines.append(class_cmidrules)
    lines.append(metric_header)
    lines.append(r"\midrule")

    for final_results, metadata in all_method_results:
        method = metadata.get("method", "unknown")
        if our_method and method == our_method:
            lines.append(r"\midrule")

        method_cell = f"\\textbf{{{method}}}" if method == our_method else method
        cells = [method_cell]
        for i, mdl in enumerate(models_present):
            for cls_name, _ in _CLASS_LATEX_ORDER:
                r = final_results.get(mdl, {}).get(cls_name, {})
                for metric in ("accuracy", "f1"):
                    mean = r.get(f"{metric}_mean", float("nan"))
                    std = r.get(f"{metric}_std", float("nan"))
                    cells.append(_decorate(mdl, cls_name, metric, method, mean, std))
            if i < n - 1:
                cells.append("")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table saved to: {out_path}")


def _print_baseline_latex_table(
    all_method_results: list,
    label_type: str,
    out_path: Path,
) -> None:
    if not all_method_results:
        return

    first_results, _ = all_method_results[0]
    baseline_order = (
        "baseline_all_zero",
        "baseline_all_one",
        "baseline_train_prevalence",
        "baseline_random_prevalence",
    )
    baseline_display = {
        "baseline_all_zero": "All zero",
        "baseline_all_one": "All one",
        "baseline_train_prevalence": "Train prevalence",
        "baseline_random_prevalence": "Random prevalence",
    }

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Baseline performance on the scRNAseq track "
        rf"(\textbf{{label type:}} {label_type}). Values are mean $\pm$ std over 5 CV folds.}}"
    )
    lines.append(r"\label{tab:scrnaseq-baselines}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l ccc ccc ccc ccc}")
    lines.append(r"\toprule")
    lines.append(
        r"& \multicolumn{3}{c}{Cl} & \multicolumn{3}{c}{ST} & "
        r"\multicolumn{3}{c}{MT} & \multicolumn{3}{c}{A} \\"
    )
    lines.append(
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} "
        r"\cmidrule(lr){8-10} \cmidrule(lr){11-13}"
    )
    lines.append(
        r"Baseline & Acc & F1 & AUPRC & Acc & F1 & AUPRC & "
        r"Acc & F1 & AUPRC & Acc & F1 & AUPRC \\"
    )
    lines.append(r"\midrule")

    for baseline in baseline_order:
        if baseline not in first_results:
            continue
        cells = [baseline_display[baseline]]
        for cls_name, _ in _CLASS_LATEX_ORDER:
            r = first_results[baseline].get(cls_name, {})
            for metric in ("accuracy", "f1", "auprc"):
                cells.append(_fmt_latex_val(
                    r.get(f"{metric}_mean", float("nan")),
                    r.get(f"{metric}_std", float("nan")),
                ))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Baseline LaTeX table saved to: {out_path}")


def _write_summary_csv(all_method_results: list, label_type: str, out_path: Path) -> None:
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auprc",
        "auc",
        "precision",
        "recall",
    )
    summary_metric_names = (
        "macro_accuracy",
        "macro_balanced_accuracy",
        "macro_f1",
        "macro_auprc",
        "macro_auc",
        "macro_precision",
        "macro_recall",
        "micro_f1",
        "micro_auprc",
        "micro_auc",
        "micro_precision",
        "micro_recall",
        "exact_match",
    )

    fieldnames = [
        "label_type",
        "method",
        "model",
        "class",
        "metric",
        "mean",
        "std",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for final_results, metadata in all_method_results:
            method = metadata.get("method", "unknown")
            for model_name, model_results in final_results.items():
                for cls_name in _CLASS_DISPLAY:
                    if cls_name not in model_results:
                        continue
                    for metric in metric_names:
                        writer.writerow({
                            "label_type": label_type,
                            "method": method,
                            "model": model_name,
                            "class": cls_name,
                            "metric": metric,
                            "mean": model_results[cls_name].get(f"{metric}_mean", float("nan")),
                            "std": model_results[cls_name].get(f"{metric}_std", float("nan")),
                        })
                summary = model_results.get("__summary__", {})
                for metric in summary_metric_names:
                    writer.writerow({
                        "label_type": label_type,
                        "method": method,
                        "model": model_name,
                        "class": "__summary__",
                        "metric": metric,
                        "mean": summary.get(f"{metric}_mean", float("nan")),
                        "std": summary.get(f"{metric}_std", float("nan")),
                    })

    print(f"CSV summary saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run scShapeBench on all methods under submissions/scRNAseq/."
    )
    parser.add_argument(
        "--submissions-dir",
        default="submissions/scRNAseq",
        help="Directory containing one subdirectory per method",
    )
    parser.add_argument(
        "--label-type",
        choices=["majority", "confidence_score","union", "high_support"],
        default="confidence_score",
    )
    parser.add_argument("--output", default="results/", help="Directory to save per-method results")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--methods",
        nargs="*",
        help="Run only these method names (default: all subdirs)",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Path to a predefined folds JSON (e.g. splits/folds_v1.json). "
             "If omitted, folds are generated randomly (seed=42).",
    )
    parser.add_argument(
        "--our-method",
        default=None,
        help="Method name to separate with \\midrule and bold in the LaTeX table",
    )
    args = parser.parse_args()

    submissions_root = Path(args.submissions_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if not submissions_root.exists():
        print(f"Submissions directory not found: {submissions_root}")
        return

    method_dirs = sorted(d for d in submissions_root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if args.methods:
        method_dirs = [d for d in method_dirs if d.name in args.methods]

    if not method_dirs:
        print(f"No method directories found under {submissions_root}")
        return

    print(f"Found {len(method_dirs)} method(s): {[d.name for d in method_dirs]}")
    print(f"Label type: {args.label_type}")

    splits_path = Path(args.splits) if args.splits else None

    all_method_results = run_real_collection(
        submissions_root,
        label_type=args.label_type,
        output_dir=output_dir,
        device=device,
        methods=args.methods,
        splits_path=splits_path,
    )

    _print_comparison_table(all_method_results, args.label_type)
    latex_path = output_dir / f"table_{args.label_type}.txt"
    _print_latex_table(all_method_results, args.label_type, latex_path, our_method=args.our_method)
    baseline_latex_path = output_dir / f"table_{args.label_type}_baselines.txt"
    _print_baseline_latex_table(all_method_results, args.label_type, baseline_latex_path)
    summary_csv_path = output_dir / f"summary_{args.label_type}.csv"
    _write_summary_csv(all_method_results, args.label_type, summary_csv_path)
    print(f"Individual result files saved to: {output_dir}")


if __name__ == "__main__":
    main()
