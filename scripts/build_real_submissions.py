"""Build real-track submissions (scReebTower, PAGA, Mapper) from scRNAseq .h5ad files.

For every ``<data-dir>/SCD-XXXX.h5ad`` and every method this writes::

    <output-dir>/<method>/SCD-XXXX/nodes.csv        node,x,y
    <output-dir>/<method>/SCD-XXXX/edges.csv        source,target,length
    <output-dir>/<method>/SCD-XXXX/h0_finite.csv    birth,death,raw_death
    <output-dir>/<method>/SCD-XXXX/h1_essential.csv birth,death,raw_birth

which is the layout ``run_benchmark.py --track real --real-mode all`` reads.

Methods run on the cell embedding in ``obsm[--rep]`` (default ``X_pca``) and
reuse the synthetic-track builders, so parameters mean the same thing on both
tracks. Node ``x, y`` are the node centroids' first two embedding coordinates,
min-max scaled to [0, 1] over the cells. Persistence uses the edge-length
filtration with lengths divided by the graph's longest edge, so ``death`` and
``birth`` lie in [0, 1] as ``features/persistence.py`` expects; the ``raw_*``
columns keep the unscaled lengths.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scshapebench.synthetic.evaluation import compute_reduced_multigraph_persistence_diagrams  # noqa: E402
from scshapebench.synthetic.method_benchmark import run_methods  # noqa: E402

METHODS = ("screebtower", "paga", "mapper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/scRNAseq", help="Directory of SCD-XXXX.h5ad files.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions/scRNAseq", help="Submissions root.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--ids", nargs="+", default=None, help="Only these dataset IDs (default: every .h5ad).")
    parser.add_argument("--rep", default="X_pca", help="obsm key holding the cell embedding.")
    parser.add_argument("--n-dims", type=int, default=None, help="Use only the first N embedding dimensions.")
    parser.add_argument("--force", action="store_true", help="Rebuild graphs that already exist.")
    parser.add_argument("--screebtower-k", type=int, default=15, help="k for the scReebTower kNN affinity.")
    parser.add_argument(
        "--screebtower-filter-method",
        default="diffusion_eigenfunction",
        choices=["diffusion_eigenfunction", "rooted_potential_distance", "rooted_diffusion_shortest_path"],
    )
    parser.add_argument("--paga-knn", type=int, default=15)
    parser.add_argument("--paga-resolution", type=float, default=1.0)
    parser.add_argument("--paga-connectivity-threshold", type=float, default=0.0)
    parser.add_argument("--mapper-cubes", type=int, default=6)
    parser.add_argument("--mapper-overlap", type=float, default=0.35)
    parser.add_argument("--mapper-min-samples", type=int, default=3)
    parser.add_argument("--mapper-eps-scale", type=float, default=1.5)
    return parser.parse_args()


def load_embedding(path: Path, rep: str, n_dims: int | None) -> np.ndarray:
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        if rep not in adata.obsm:
            raise ValueError(f"{path.name}: obsm[{rep!r}] not found (have {list(adata.obsm.keys())}).")
        points = np.asarray(adata.obsm[rep], dtype=float)
    finally:
        adata.file.close()
    return points[:, :n_dims] if n_dims else points


def build_screebtower_graph(points: np.ndarray, args: argparse.Namespace):
    if run_methods.screebtower is None:
        raise SystemExit(
            'scReebTower is not installed: pip install "sc-reeb-tower @ git+https://github.com/KrishnaswamyLab/scReebTower"'
        )
    result = run_methods.screebtower(
        X=points,
        k=int(args.screebtower_k),
        precision=1.0,
        filter_method=args.screebtower_filter_method,
        simplify=True,
        return_edge_lengths=True,
        return_point_assignment=True,
    )
    graph = result["simplified"]
    assignment = np.asarray(graph["point_assignment"], dtype=int)
    positions = {}
    for node in range(len(graph["nodes"])):
        members = assignment == node
        positions[node] = points[members].mean(axis=0) if members.any() else points.mean(axis=0)
    edges = [(int(u), int(v), float(length)) for (u, v), length in zip(graph["edges"], graph["edge_lengths"])]
    return positions, edges


def build_simple_graph(method: str, points: np.ndarray, args: argparse.Namespace):
    builder = run_methods.build_paga_payload if method == "paga" else run_methods.build_mapper_payload
    payload = builder(points, args)
    positions = {int(node): np.asarray(value, dtype=float) for node, value in payload["node_positions"].items()}
    edges = [
        (int(u), int(v), float(np.linalg.norm(positions[int(u)] - positions[int(v)])))
        for u, v in payload["edges"]
    ]
    return positions, edges


def write_graph(sample_dir: Path, positions: dict, edges: list, points: np.ndarray) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    lo = points[:, :2].min(axis=0)
    span = np.where(np.ptp(points[:, :2], axis=0) > 0, np.ptp(points[:, :2], axis=0), 1.0)

    with open(sample_dir / "nodes.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node", "x", "y"])
        for node in sorted(positions):
            x, y = (positions[node][:2] - lo) / span
            writer.writerow([node, float(x), float(y)])

    with open(sample_dir / "edges.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "length"])
        writer.writerows(edges)

    # The real-track scorer (features/persistence.py) expects values divided by the
    # graph's longest edge, i.e. in [0, 1]; the synthetic track's median scaling
    # would push values past 1, where the persistence images drop them.
    lengths = [length for _, _, length in edges if np.isfinite(length) and length > 0]
    scale = max(lengths) if lengths else 1.0
    multigraph = nx.MultiGraph()
    multigraph.add_nodes_from(positions)
    for u, v, length in edges:
        multigraph.add_edge(u, v, length=length / scale)
    diagrams = compute_reduced_multigraph_persistence_diagrams(multigraph, normalize_edge_lengths=False)

    with open(sample_dir / "h0_finite.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["birth", "death", "raw_death"])
        writer.writerows([birth, death, death * scale] for birth, death in diagrams["h0_finite"])

    with open(sample_dir / "h1_essential.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["birth", "death", "raw_birth"])
        writer.writerows([birth, "inf", birth * scale] for birth, _ in diagrams["h1_essential"])


def main() -> None:
    args = parse_args()
    paths = sorted(args.data_dir.glob("*.h5ad"))
    if args.ids:
        wanted = set(args.ids)
        paths = [path for path in paths if path.stem in wanted]
        missing = wanted - {path.stem for path in paths}
        if missing:
            raise SystemExit(f"No .h5ad for: {', '.join(sorted(missing))} in {args.data_dir}")
    if not paths:
        raise SystemExit(f"No .h5ad files in {args.data_dir}")

    for path in paths:
        sample_id = path.stem
        todo = [m for m in args.methods if args.force or not (args.output_dir / m / sample_id / "edges.csv").is_file()]
        if not todo:
            print(f"[{sample_id}] cached", flush=True)
            continue
        points = load_embedding(path, args.rep, args.n_dims)
        print(f"[{sample_id}] {points.shape[0]} cells x {points.shape[1]} dims", flush=True)
        for method in todo:
            if method == "screebtower":
                positions, edges = build_screebtower_graph(points, args)
            else:
                positions, edges = build_simple_graph(method, points, args)
            write_graph(args.output_dir / method / sample_id, positions, edges, points)
            print(f"  - {method}: {len(positions)} nodes, {len(edges)} edges", flush=True)


if __name__ == "__main__":
    main()
