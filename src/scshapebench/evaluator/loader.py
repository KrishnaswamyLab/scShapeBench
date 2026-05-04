import csv
import json
import re
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
from torch_geometric.data import Data


def _load_graph(sample_dir: Path) -> Data:
    """Load nodes.csv + edges.csv from a sample directory into a PyG Data object."""
    nodes_path = sample_dir / "nodes.csv"
    edges_path = sample_dir / "edges.csv"

    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes.csv in {sample_dir}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges.csv in {sample_dir}")

    with open(nodes_path) as f:
        nodes = [(float(r["x"]), float(r["y"])) for r in csv.DictReader(f)]
    coords = torch.tensor(nodes, dtype=torch.float)

    src, dst, lengths = [], [], []
    with open(edges_path) as f:
        for row in csv.DictReader(f):
            src.append(int(row["source"]))
            dst.append(int(row["target"]))
            lengths.append(float(row["length"]) if row.get("length") else 0.0)

    # Bidirectional edges
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    edge_weight = torch.tensor(lengths + lengths, dtype=torch.float)
    if edge_weight.numel() > 0:
        edge_weight = edge_weight / edge_weight.max().clamp(min=1e-8)

    g = Data(x=coords, edge_index=edge_index, edge_weight=edge_weight)

    h0_path = sample_dir / "h0_finite.csv"
    if h0_path.exists():
        with open(h0_path) as f:
            h0_deaths = [float(r["death"]) for r in csv.DictReader(f)]
        g.h0_deaths = torch.tensor(h0_deaths, dtype=torch.float)

    h1_path = sample_dir / "h1_essential.csv"
    if h1_path.exists():
        with open(h1_path) as f:
            h1_births = [float(r["birth"]) for r in csv.DictReader(f)]
        g.h1_births = torch.tensor(h1_births, dtype=torch.float)

    return g


def _parse_sample_id(name: str) -> int | None:
    """Extract trailing integer from a directory name (e.g. '42' or 'SCD-0042')."""
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


def load_graphs(submission_dir: Path) -> Tuple[Dict[str, Data], Dict[str, Any]]:
    """
    Load graphs from a submission directory.

    Expected structure:
        submission_dir/
        ├── SC-001/
        │   ├── nodes.csv   (columns: id, x, y)
        │   └── edges.csv   (columns: source, target, length)
        ├── SC-002/
        ...

    Sample directory names are normalised to the benchmark ID format
    ``SCD-XXXX`` (e.g. ``SC-001`` → ``SCD-0001``) so they match the
    predefined splits in ``splits/folds_v1.json``.
    """
    if not submission_dir.exists():
        raise FileNotFoundError(f"Submission directory not found: {submission_dir}")

    sample_dirs = sorted(
        [d for d in submission_dir.iterdir() if d.is_dir()],
        key=lambda d: (_parse_sample_id(d.name) if _parse_sample_id(d.name) is not None else -1),
    )

    if not sample_dirs:
        raise ValueError(f"No sample subdirectories found in {submission_dir}")

    graphs: Dict[str, Data] = {}
    print(f"Loading {len(sample_dirs)} graphs from {submission_dir}...")

    for sample_dir in sample_dirs:
        num = _parse_sample_id(sample_dir.name)
        key = f"SCD-{num:04d}" if num is not None else sample_dir.name
        graphs[key] = _load_graph(sample_dir)

    metadata: Dict[str, Any] = {
        "submission_dir": str(submission_dir),
        "method": submission_dir.name,
        "n_graphs": len(graphs),
    }

    print(f"Loaded {len(graphs)} graphs (IDs: {sorted(graphs)[:5]}...)")
    return graphs, metadata
