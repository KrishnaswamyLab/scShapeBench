"""Print real-track results in the paper's per-class Acc / F1 layout.

Reads the ``summary_<label_type>.csv`` written by ``run_benchmark.py --track real
--real-mode all`` and prints one row per method, with mean±std over the 5 folds for
accuracy and F1 per shape class (Cl, ST, MT, A) for each chosen model::

    GNN MLP
    Cl ST MT A Cl ST MT A
    Method Acc F1 Acc F1 Acc F1 Acc F1 Acc F1 Acc F1 Acc F1 Acc F1
    mapper .61±.08 .71±.07 ...

The paper's table uses ``--label-type union``. Example::

    python run_benchmark.py --track real --real-mode all \\
      --submissions-dir submissions/scRNAseq --label-type union \\
      --splits splits/folds_v1.json --output results/union --device cpu
    python scripts/paper_table.py results/union/summary_union.csv --methods mapper paga
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

CLASSES = [("clusters", "Cl"), ("simple_traj", "ST"), ("multi_branch", "MT"), ("archetypal", "A")]
METRICS = [("accuracy", "Acc"), ("f1", "F1")]
MODEL_NAMES = {"gnn": "GNN", "mlp": "MLP", "pi_svm": "PI-SVM", "stats_svm": "Stats-SVM", "rf": "RF"}


def fmt(value: float) -> str:
    """Two decimals without the leading zero, as in the paper (0.613 -> .61)."""
    text = f"{value:.2f}"
    return text[1:] if text.startswith("0.") else text


def load_summary(path: Path) -> dict[tuple[str, str, str, str], tuple[float, float]]:
    with open(path, newline="") as handle:
        return {
            (row["method"], row["model"], row["class"], row["metric"]): (float(row["mean"]), float(row["std"]))
            for row in csv.DictReader(handle)
        }


def paper_rows(summary: dict, methods: list[str], models: list[str]) -> list[list[str]]:
    """One list of cells per method: [method, 'm±s', ...] in model × class × metric order."""
    rows = []
    for method in methods:
        cells = [method]
        for model in models:
            for cls, _ in CLASSES:
                for metric, _ in METRICS:
                    key = (method, model, cls, metric)
                    if key not in summary:
                        raise SystemExit(f"Missing {key} in the summary CSV")
                    mean, std = summary[key]
                    cells.append(f"{fmt(mean)}±{fmt(std)}")
        rows.append(cells)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary_csv", type=Path, help="summary_<label_type>.csv from run_benchmark.py")
    parser.add_argument("--methods", nargs="+", default=None, help="Rows, in order (default: all methods in the CSV).")
    parser.add_argument("--models", nargs="+", default=["gnn", "mlp"], choices=list(MODEL_NAMES))
    parser.add_argument("--format", choices=["text", "markdown"], default="text")
    args = parser.parse_args()

    summary = load_summary(args.summary_csv)
    methods = args.methods or list(dict.fromkeys(key[0] for key in summary))
    rows = paper_rows(summary, methods, args.models)

    class_header = [short for _, short in CLASSES] * len(args.models)
    metric_header = [short for _, short in METRICS] * len(CLASSES) * len(args.models)
    if args.format == "text":
        print(" ".join(MODEL_NAMES[m] for m in args.models))
        print(" ".join(class_header))
        print(" ".join(["Method", *metric_header]))
        for row in rows:
            print(" ".join(row))
    else:
        header = ["Method"] + [
            f"{MODEL_NAMES[m]} {c} {k}" for m in args.models for _, c in CLASSES for _, k in METRICS
        ]
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for row in rows:
            print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
