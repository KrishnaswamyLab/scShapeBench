# -*- coding: utf-8 -*-
"""
Generate stratified k-fold splits from the aggregated label files.

Usage
-----
python scripts/make_splits.py [--labels-file FILE] [--n-splits N] [--seed S] [--output PATH]

Reads majority_labels.json (produced by make_labels.py) by default.
Splits are stratified by dominant class (argmax of the label vector).
Output splits use dataset names (e.g. "SCD-0042"), not integer indices.
"""

import argparse
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_LABELS = REPO_ROOT / "benchmarks" / "scrnaseq" / "labels" / "majority_labels.json"
DEFAULT_OUTPUT = REPO_ROOT / "splits" / "folds_v1.json"


def load_datasets(labels_path: Path) -> tuple[list[str], list[int]]:
    with open(labels_path) as f:
        data = json.load(f)

    dataset_order = data["dataset_order"]
    label_matrix = data["labels"]

    ids: list[str] = []
    stratify_labels: list[int] = []

    for idx, ds_id in enumerate(dataset_order):
        vec = label_matrix[str(idx)]
        ids.append(ds_id)
        # Use dominant class index for stratification; ties broken by first max.
        stratify_labels.append(int(max(range(len(vec)), key=lambda i: vec[i])))

    return ids, stratify_labels


def create_splits(ids: list[str], labels: list[int], n_splits: int, seed: int) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = {}
    for fold, (train_idx, test_idx) in enumerate(skf.split(ids, labels)):
        splits[f"fold_{fold}"] = {
            "train": sorted([ids[i] for i in train_idx]),
            "test": sorted([ids[i] for i in test_idx]),
        }
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stratified k-fold splits from aggregated label files."
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=DEFAULT_LABELS,
        help=f"Path to labels JSON (default: {DEFAULT_LABELS})",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    ids, labels = load_datasets(args.labels_file)
    print(f"Loaded {len(ids)} datasets from {args.labels_file}")

    splits = create_splits(ids, labels, n_splits=args.n_splits, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved {args.n_splits}-fold splits -> {args.output}")
    for fold, split in splits.items():
        print(f"  {fold}: {len(split['train'])} train / {len(split['test'])} test")


if __name__ == "__main__":
    main()
