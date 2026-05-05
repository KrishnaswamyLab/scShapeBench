"""
Hand-crafted graph structural features extracted from a submission graph S.

No access to the original point cloud — only graph topology and edge weights are used.

Feature groups (16 total):
  (i)   Size stats (3):         n_nodes, n_edges, n_components
  (ii)  Degree distribution (5): mean, variance, max, frac_leaves, frac_branch_points
  (iii) Cyclomatic stats (3):   cycle_rank = |E|−|V|+β₀, n_independent_cycles, mean_cycle_length
  (iv)  Distance stats (2):     diameter/n, mean_shortest_path/n  (largest component, hop count)
  (v)   Edge-weight stats (3):  mean_weight, var_weight, max_to_mean_ratio
"""

from typing import List

import networkx as nx
import numpy as np
from torch_geometric.data import Data

FEAT_DIM = 16


def _graph_to_features(g: Data) -> np.ndarray:
    n_nodes  = g.x.size(0)
    src      = g.edge_index[0].numpy()
    dst      = g.edge_index[1].numpy()
    raw_w    = g.edge_weight.numpy() if g.edge_weight.numel() > 0 else None

    # Build undirected graph; PyG stores each edge twice — deduplicate.
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    seen = set()
    for idx, (u, v) in enumerate(zip(src, dst)):
        if u == v:
            continue
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if key not in seen:
            seen.add(key)
            w = float(raw_w[idx]) if raw_w is not None else 1.0
            G.add_edge(*key, weight=w)

    components  = list(nx.connected_components(G))
    n_comp      = len(components)
    n_nodes_g   = G.number_of_nodes()
    n_edges_g   = G.number_of_edges()

    # (i) Size
    size_feats = [float(n_nodes_g), float(n_edges_g), float(n_comp)]

    # (ii) Degree distribution
    degrees = [d for _, d in G.degree()]
    if degrees:
        mean_deg    = float(np.mean(degrees))
        var_deg     = float(np.var(degrees))
        max_deg     = float(max(degrees))
        frac_leaves = sum(1 for d in degrees if d == 1) / n_nodes_g
        frac_branch = sum(1 for d in degrees if d >= 3) / n_nodes_g
    else:
        mean_deg = var_deg = max_deg = frac_leaves = frac_branch = 0.0
    deg_feats = [mean_deg, var_deg, max_deg, frac_leaves, frac_branch]

    # (iii) Cyclomatic stats
    cycle_rank = n_edges_g - n_nodes_g + n_comp  # = dim H_1 for undirected graph
    all_cycles: list = []
    for comp in components:
        all_cycles.extend(nx.cycle_basis(G.subgraph(comp)))
    n_cycles       = len(all_cycles)
    mean_cycle_len = float(np.mean([len(c) for c in all_cycles])) if all_cycles else 0.0
    cyclo_feats    = [float(cycle_rank), float(n_cycles), mean_cycle_len]

    # (iv) Distance stats on largest component (unweighted hop count, normalized by n)
    largest = max(components, key=len)
    subG    = G.subgraph(largest)
    n_large = len(largest)
    if n_large > 1:
        try:
            diam      = nx.diameter(subG)
            pair_lens = dict(nx.all_pairs_shortest_path_length(subG))
            total     = sum(v for row in pair_lens.values() for v in row.values())
            n_pairs   = n_large * (n_large - 1)
            mean_path = total / n_pairs if n_pairs > 0 else 0.0
            diam_norm = diam     / n_large
            path_norm = mean_path / n_large
        except nx.NetworkXError:
            diam_norm = path_norm = 0.0
    else:
        diam_norm = path_norm = 0.0
    dist_feats = [diam_norm, path_norm]

    # (v) Edge-weight summaries
    w_vals = [d["weight"] for _, _, d in G.edges(data=True)]
    if w_vals:
        mean_w      = float(np.mean(w_vals))
        var_w       = float(np.var(w_vals))
        max_w       = float(max(w_vals))
        max_to_mean = max_w / (mean_w + 1e-8)
    else:
        mean_w = var_w = max_to_mean = 0.0
    weight_feats = [mean_w, var_w, max_to_mean]

    return np.array(size_feats + deg_feats + cyclo_feats + dist_feats + weight_feats,
                    dtype=np.float32)


def extract_features(
    train_graphs: List[Data],
    test_graphs:  List[Data],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X_train : [N_train, FEAT_DIM]
    X_test  : [N_test,  FEAT_DIM]
    """
    X_train = np.stack([_graph_to_features(g) for g in train_graphs])
    X_test  = np.stack([_graph_to_features(g) for g in test_graphs])
    return X_train, X_test
