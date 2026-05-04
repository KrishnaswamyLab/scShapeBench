from typing import Dict, Any

import torch
from torch_geometric.data import Data


def validate_submission(graphs: Dict[str, Data]) -> None:
    if not graphs:
        raise ValueError("Submission contains no graphs.")

    for sample_id, g in graphs.items():
        if not hasattr(g, "edge_index") or g.edge_index is None:
            raise ValueError(f"Sample {sample_id}: missing 'edge_index'.")

        if g.edge_index.dtype != torch.long:
            raise TypeError(
                f"Sample {sample_id}: edge_index must be LongTensor, got {g.edge_index.dtype}."
            )

        if g.edge_index.dim() != 2 or g.edge_index.shape[0] != 2:
            raise ValueError(
                f"Sample {sample_id}: edge_index must have shape [2, E], got {list(g.edge_index.shape)}."
            )

        if g.x is not None:
            if torch.isnan(g.x).any():
                raise ValueError(f"Sample {sample_id}: NaN in node features (x).")
            if torch.isinf(g.x).any():
                raise ValueError(f"Sample {sample_id}: Inf in node features (x).")

        if hasattr(g, "edge_weight") and g.edge_weight is not None:
            if torch.isnan(g.edge_weight).any():
                raise ValueError(f"Sample {sample_id}: NaN in edge weights.")

    print(f"Validation passed: {len(graphs)} graphs.")
