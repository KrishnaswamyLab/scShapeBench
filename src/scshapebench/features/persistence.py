"""
Persistence barcode features extracted from loaded graph objects.

Each graph stores h0_deaths and h1_births tensors (loaded from h0_finite.csv
and h1_essential.csv). This module converts them to a fixed 18-dim vector
using the same statistics as the Reeb-Classifier benchmark.
"""

from typing import List

import numpy as np
from torch_geometric.data import Data

FEAT_DIM = 18  # 9 H0 stats + 9 H1 stats


def _persistence_stats(vals: np.ndarray) -> np.ndarray:
    if len(vals) == 0:
        return np.zeros(9, dtype=np.float32)
    total = vals.sum()
    entropy = 0.0
    if total > 0:
        p = vals / total
        entropy = float(-np.sum(p * np.log(p + 1e-10)))
    return np.array([
        len(vals),
        float(total),
        float(vals.max()),
        float(vals.mean()),
        float(vals.std()),
        entropy,
        float(np.percentile(vals, 25)),
        float(np.percentile(vals, 50)),
        float(np.percentile(vals, 75)),
    ], dtype=np.float32)


def _graph_to_features(g: Data) -> np.ndarray:
    h0 = g.h0_deaths.numpy() if hasattr(g, "h0_deaths") else np.array([], dtype=np.float32)
    h1 = g.h1_births.numpy() if hasattr(g, "h1_births") else np.array([], dtype=np.float32)
    return np.concatenate([_persistence_stats(h0), _persistence_stats(h1)])


def extract_features(
    train_graphs: List[Data],
    test_graphs: List[Data],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute raw persistence feature matrices (unnormalized).
    Each model handles its own normalization to avoid data leakage.

    Returns:
        X_train: [N_train, FEAT_DIM]
        X_test:  [N_test,  FEAT_DIM]
    """
    X_train = np.stack([_graph_to_features(g) for g in train_graphs])
    X_test  = np.stack([_graph_to_features(g) for g in test_graphs])
    return X_train, X_test
