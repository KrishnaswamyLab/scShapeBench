"""
Persistence image features extracted from graph homology data.

Each graph stores:
  h0_deaths  — finite H0 deaths, normalized to [0, 1] by the graph's max edge weight
  h1_births  — essential H1 births, normalized to [0, 1] (death is always inf in a graph filtration)

We build (birth, persistence) pairs:
  H0: (0, death)        — components merge at the edge weight 'death'
  H1: (birth, 1−birth)  — essential cycles; death is capped at the normalized maximum (1.0)

Each pair set is vectorized as a persistence image (Adams et al. 2017):
  weighted sum of 2D Gaussians on a fixed RESOLUTION × RESOLUTION grid in [0,1]²,
  with linear persistence weighting  w(b, p) = p.

FEAT_DIM = 2 × RESOLUTION² (H0 image ‖ H1 image, both flattened row-major).
"""

from typing import List

import numpy as np
from torch_geometric.data import Data

RESOLUTION = 10      # pixels per axis per homology group
SIGMA      = 0.1     # Gaussian bandwidth (values live in [0, 1])
FEAT_DIM   = 2 * RESOLUTION * RESOLUTION  # 200

# Pre-compute the evaluation grid once (shared across all calls).
_GRID    = np.linspace(0.0, 1.0, RESOLUTION, dtype=np.float32)
_BB, _PP = np.meshgrid(_GRID, _GRID)  # birth-axis, persistence-axis; each [R, R]
_INV2S2  = 1.0 / (2.0 * SIGMA ** 2)


def _persistence_image(pairs: np.ndarray) -> np.ndarray:
    """
    pairs: [N, 2] float32 array of (birth, persistence) values in [0, 1]².
    Returns a flattened [RESOLUTION²] persistence image.
    """
    img = np.zeros((RESOLUTION, RESOLUTION), dtype=np.float32)
    for b, p in pairs:
        if p <= 0.0:
            continue
        img += p * np.exp(-((_BB - b) ** 2 + (_PP - p) ** 2) * _INV2S2)
    return img.flatten()


def _graph_to_features(g: Data) -> np.ndarray:
    # H0: birth = 0, persistence = death
    if hasattr(g, "h0_deaths") and g.h0_deaths.numel() > 0:
        deaths   = g.h0_deaths.numpy().astype(np.float32)
        h0_pairs = np.column_stack([np.zeros(len(deaths), dtype=np.float32), deaths])
    else:
        h0_pairs = np.empty((0, 2), dtype=np.float32)

    # H1: persistence = 1 − birth  (essential bars, death capped at 1)
    if hasattr(g, "h1_births") and g.h1_births.numel() > 0:
        births   = g.h1_births.numpy().astype(np.float32)
        pers     = np.clip(1.0 - births, 0.0, 1.0)
        h1_pairs = np.column_stack([births, pers])
    else:
        h1_pairs = np.empty((0, 2), dtype=np.float32)

    return np.concatenate([
        _persistence_image(h0_pairs),
        _persistence_image(h1_pairs),
    ])


def extract_features(
    train_graphs: List[Data],
    test_graphs:  List[Data],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute persistence image feature matrices (unnormalized).
    Each downstream model handles its own z-score normalization to avoid data leakage.

    Returns
    -------
    X_train : [N_train, FEAT_DIM]
    X_test  : [N_test,  FEAT_DIM]
    """
    X_train = np.stack([_graph_to_features(g) for g in train_graphs])
    X_test  = np.stack([_graph_to_features(g) for g in test_graphs])
    return X_train, X_test
