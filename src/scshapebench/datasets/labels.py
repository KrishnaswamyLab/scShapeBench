import json
from pathlib import Path
from typing import Tuple
import os
import numpy as np

_DEFAULT_LABELS_PATH = os.path.join("benchmarks","scrnaseq","labels","union_labels.json")

CLASS_ORDER = ["archetypal", "multi_branch", "simple_traj", "clusters"]

LABEL_MAP: dict[str, str] = {
    "clusters": "clusters",
    "simple_traj": "simple_traj",
    "cyclic": "simple_traj",
    "bifurcation": "multi_branch",
    "multi_branch": "multi_branch",
    "complex_tree": "multi_branch",
    "archetypal": "archetypal",
    "surface": "archetypal",
}


def load_labels(path: str | None = None) -> Tuple[np.ndarray, list[str], list[str]]:
    """
    Load aggregated labels produced by scripts/make_labels.py.

    Returns
    -------
    labels : float32 array of shape [n_datasets, n_classes]
        Row i corresponds to dataset_order[i].
    class_order : list of class name strings, length n_classes
    dataset_order : list of dataset ID strings (e.g. "SCD-0001"), length n_datasets
    """
    labels_path = Path(path) if path else _DEFAULT_LABELS_PATH

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with open(labels_path) as f:
        data = json.load(f)

    class_order: list[str] = data["class_order"]
    dataset_order: list[str] = data["dataset_order"]
    raw: dict[str, list[float]] = data["labels"]

    n = len(dataset_order)
    k = len(class_order)
    labels = np.zeros((n, k), dtype=np.float32)
    for idx_str, vec in raw.items():
        labels[int(idx_str)] = vec

    return labels, class_order, dataset_order
