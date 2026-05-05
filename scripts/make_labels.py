# -*- coding: utf-8 -*-
"""
Aggregate per-annotator annotation files into benchmarks/scrnaseq/labels/.

Usage
-----
python scripts/make_labels.py [--min-annotators N]

Output files (all written to benchmarks/scrnaseq/labels/)
----------------------------------------------------------
majority_labels.json     -- label present if >50% of annotators assigned it (hard binary)
union_labels.json        -- label present if >=1 annotator assigned it (broadest coverage)
soft_labels.json         -- fraction of annotators that assigned each label (0.0-1.0)
confidence_labels.json   -- per-class confidence-weighted majority vote:
                            S_k = sum_i c_i * 1{k in L_i}; active if S_k > 0.5 * sum_i c_i

The script only includes SCD-XXXX datasets and skips any dataset annotated by fewer
than --min-annotators annotators (default: 5).

Dataset index assignment
------------------------
Datasets are sorted numerically by their trailing integer (SCD-0001 -> index 0,
SCD-0002 -> index 1, etc.), which matches the integer indices used in splits/*.json.

Label mapping
-------------
Raw annotation labels are collapsed into 4 canonical classes before aggregation:
  clusters     <- clusters
  simple_traj  <- simple_traj, cyclic
  multi_branch <- bifurcation, multi_branch, complex_tree
  archetypal   <- archetypal, surface
Labels not listed (blob, batch_effect, outlier_dominated) are ignored.
Each annotator contributes at most 1 vote per canonical class regardless of how many
raw labels they assigned that map to the same canonical.

Confidence scores
-----------------
ANNOTATOR_CONFIDENCE maps each annotator index i (matching annotation_i.json) to its
self-reported confidence weight c_i used by the confidence_weighted aggregation mode.
Update the values in that dict before running to reflect the actual reported scores.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
ANNOTATIONS_DIR = REPO_ROOT / "benchmarks" / "scrnaseq" / "annotations"
OUTPUT_DIR = REPO_ROOT / "benchmarks" / "scrnaseq" / "labels"

sys.path.insert(0, str(REPO_ROOT / "src"))
from scshapebench.datasets.labels import CLASS_ORDER, LABEL_MAP  # noqa: E402

# Self-reported confidence scores, one per annotator file (annotation_i.json -> key i).
# Replace placeholder values with the actual scores collected from annotators.
ANNOTATOR_CONFIDENCE: dict[int, float] = {
    0: 5.0,
    1: 3.0,
    2: 3.0,
    3: 1.0,
    4: 3.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
}

MODE_FILENAME = {
    "majority": "majority_labels.json",
    "union": "union_labels.json",
    "soft": "soft_labels.json",
    "confidence_weighted": "confidence_labels.json",
}


def load_all_annotations(annotations_dir: Path) -> dict[str, list[tuple[int, list[str]]]]:
    """Return {dataset_id: [(annotator_idx, labels), ...]}."""
    per_dataset: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    for ann_idx, path in enumerate(sorted(annotations_dir.glob("annotation_*.json"))):
        with open(path) as f:
            data = json.load(f)
        for dataset_id, labels in data["labels"].items():
            per_dataset[dataset_id].append((ann_idx, labels))
    return dict(per_dataset)


def map_labels(raw_labels: list[str]) -> set[str]:
    """Map a single annotator's raw labels to canonical class names, dropping unknowns."""
    return {LABEL_MAP[l] for l in raw_labels if l in LABEL_MAP}


def aggregate_labels(
    per_annotator: list[tuple[int, list[str]]],
    mode: str,
) -> list[float]:
    n = len(per_annotator)

    if mode == "confidence_weighted":
        # S_k = sum_i c_i * 1{k in L_i}; active if S_k > 0.5 * sum_i c_i
        support: dict[str, float] = defaultdict(float)
        total_weight = 0.0
        for ann_idx, ann_labels in per_annotator:
            weight = ANNOTATOR_CONFIDENCE[ann_idx]
            total_weight += weight
            for canonical in map_labels(ann_labels):
                support[canonical] += weight
        threshold = 0.5 * total_weight
        return [1.0 if support[c] > threshold else 0.0 for c in CLASS_ORDER]

    counts: dict[str, int] = defaultdict(int)
    for _, ann_labels in per_annotator:
        for canonical in map_labels(ann_labels):
            counts[canonical] += 1

    if mode == "majority":
        return [1.0 if counts[c] / n > 0.5 else 0.0 for c in CLASS_ORDER]
    if mode == "union":
        return [1.0 if counts[c] > 0 else 0.0 for c in CLASS_ORDER]
    if mode == "soft":
        return [round(counts[c] / n, 4) for c in CLASS_ORDER]
    raise ValueError(f"Unknown mode: {mode!r}")


def print_summary(
    mode: str,
    dataset_order: list[str],
    label_matrix: dict[str, list[float]],
) -> None:
    n = len(dataset_order)
    print(f"\n[{mode}] {'Dataset':<12}  labels")
    print("-" * 60)
    for idx, ds_id in enumerate(dataset_order[:10]):
        vec = label_matrix[str(idx)]
        active = [CLASS_ORDER[i] for i, v in enumerate(vec) if v > 0]
        print(f"  {ds_id:<12}  {active}")
    if n > 10:
        print(f"  ... ({n - 10} more datasets)")

    print(f"\n  Class prevalence ({n} datasets):")
    for i, cls in enumerate(CLASS_ORDER):
        total = sum(label_matrix[str(j)][i] for j in range(n))
        print(f"    {cls:<22} {total:>5.1f}  ({100 * total / n:.1f}%)")

    print(f"\n  Label-count distribution ({n} datasets):")
    for k in range(len(CLASS_ORDER) + 1):
        count = sum(1 for j in range(n) if sum(v > 0 for v in label_matrix[str(j)]) == k)
        bar = "#" * count
        print(f"    {k} label(s): {count:>4}  ({100 * count / n:5.1f}%)  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build aggregated label files from per-annotator annotations."
    )
    parser.add_argument(
        "--min-annotators",
        type=int,
        default=5,
        metavar="N",
        help="Minimum annotators required to include a dataset (default: 5)",
    )
    args = parser.parse_args()

    raw = load_all_annotations(ANNOTATIONS_DIR)

    included = {
        k: v
        for k, v in raw.items()
        if k.startswith("SCD-") and len(v) >= args.min_annotators
    }

    dataset_order = sorted(included, key=lambda k: int(k.split("-")[1]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for mode, filename in MODE_FILENAME.items():
        label_matrix: dict[str, list[float]] = {
            str(idx): aggregate_labels(included[ds_id], mode)
            for idx, ds_id in enumerate(dataset_order)
        }

        out = {
            "class_order": CLASS_ORDER,
            "label_map": LABEL_MAP,
            "aggregation": mode,
            "min_annotators": args.min_annotators,
            "dataset_order": dataset_order,
            "labels": label_matrix,
        }

        output_path = OUTPUT_DIR / filename
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"Wrote {len(label_matrix)} datasets -> {output_path}")
        print_summary(mode, dataset_order, label_matrix)


if __name__ == "__main__":
    main()
