"""Tutorial: benchmark a k-nearest-neighbour graph on the real track.

Builds one graph per dataset (one node per cell, each cell joined to its k
nearest neighbours in ``obsm["X_pca"]``) and writes it in the submission layout
that ``run_benchmark.py --track real`` reads::

    <output-dir>/SCD-XXXX/{nodes,edges,h0_finite,h1_essential}.csv

Replace the block marked "your method" to benchmark your own graphs.

Run from the repository root:

    python examples/knn_submission.py --data-dir data/sample --output-dir submissions/scRNAseq/knn2
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_real_submissions import write_graph  # noqa: E402  writes the 4 CSVs on the scorer's scale

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data-dir", type=Path, default=ROOT / "data/sample", help="Directory of SCD-XXXX.h5ad files.")
parser.add_argument("--output-dir", type=Path, default=ROOT / "submissions/scRNAseq/knn2", help="Method folder to write.")
parser.add_argument("--k", type=int, default=2, help="Neighbours per cell.")
parser.add_argument("--ids", nargs="+", default=None, help="Only these dataset IDs (default: every .h5ad).")
parser.add_argument("--force", action="store_true", help="Rebuild graphs that already exist.")
args = parser.parse_args()

paths = sorted(args.data_dir.glob("SCD-*.h5ad"))
if args.ids:
    paths = [path for path in paths if path.stem in set(args.ids)]
    missing = set(args.ids) - {path.stem for path in paths}
    if missing:
        raise SystemExit(f"No .h5ad for: {', '.join(sorted(missing))} in {args.data_dir}")
if not paths:
    raise SystemExit(f"No SCD-*.h5ad files in {args.data_dir}")

for path in paths:
    if not args.force and (args.output_dir / path.stem / "h1_essential.csv").is_file():
        print(f"{path.stem}: cached")
        continue
    adata = ad.read_h5ad(path, backed="r")
    if "X_pca" not in adata.obsm:
        print(f"{path.stem}: skipped (no obsm['X_pca'])")
        continue
    points = np.asarray(adata.obsm["X_pca"], dtype=float)

    # ---- your method: any graph over the cells ---------------------------------
    # positions: {node_id: coordinates in the same space as `points`}
    # edges:     [(source_node, target_node, length), ...]
    dist, idx = NearestNeighbors(n_neighbors=args.k + 1).fit(points).kneighbors(points)
    edge_lengths = {}
    for i in range(len(points)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):  # column 0 is the cell itself
            edge_lengths[(min(i, int(j)), max(i, int(j)))] = float(d)
    positions = {i: points[i] for i in range(len(points))}
    edges = [(u, v, d) for (u, v), d in edge_lengths.items()]
    # -----------------------------------------------------------------------------

    write_graph(args.output_dir / path.stem, positions, edges, points)
    print(f"{path.stem}: {len(positions)} nodes, {len(edges)} edges")
