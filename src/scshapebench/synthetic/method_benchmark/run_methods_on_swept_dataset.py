"""Run scReebTower, PAGA, and Mapper on the new swept benchmark dataset.

This is a small convenience wrapper around ``method_benchmark/run_methods.py``.
It defaults to the thickness-aware swept dataset produced by
``generate_1000_swept_dataset.py`` and keeps the usual baseline methods enabled.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[4]
# run_methods uses package-relative imports, so it must run as a module.
RUN_METHODS_MODULE = "scshapebench.synthetic.method_benchmark.run_methods"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=str(BENCHMARK_ROOT / "data/synthetic"),
        help="Dataset directory containing sample_*.npz/.json (manifest.jsonl optional).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BENCHMARK_ROOT / "results/method_benchmark"),
        help="Directory for cached method outputs and result tables.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["screebtower", "screebtower_smooth", "paga", "mapper", "gtda_mapper"],
        help="Methods to run. Defaults to all benchmark methods.",
    )
    parser.add_argument("--start", type=int, default=0, help="First manifest row to process.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to process.")
    parser.add_argument(
        "--pdf-sample-size",
        type=int,
        default=10,
        help="Number of random samples to include in the diagnostic PDFs.",
    )
    parser.add_argument(
        "--pdf-seed",
        type=int,
        default=0,
        help="Random seed for selecting diagnostic PDF samples.",
    )
    parser.add_argument("--no-pdfs", action="store_true", help="Skip random-sample figure PDF generation.")
    parser.add_argument("--force", action="store_true", help="Regenerate cached method outputs.")
    parser.add_argument("--screebtower-k", type=int, default=15, help="k for scReebTower point-cloud affinity.")
    parser.add_argument(
        "--screebtower-filter-method",
        default="diffusion_eigenfunction",
        choices=["diffusion_eigenfunction", "rooted_potential_distance", "rooted_diffusion_shortest_path"],
        help="scReebTower scalar filter method for point-cloud outputs.",
    )
    parser.add_argument("--paga-knn", type=int, default=15, help="kNN size for PAGA-style connectivity.")
    parser.add_argument("--paga-resolution", type=float, default=1.0, help="Leiden resolution before Scanpy PAGA.")
    parser.add_argument(
        "--paga-connectivity-threshold",
        type=float,
        default=0.0,
        help="Keep PAGA edges with connectivity greater than this threshold.",
    )
    parser.add_argument("--mapper-cubes", type=int, default=6, help="Mapper cover cubes per lens dimension.")
    parser.add_argument("--mapper-overlap", type=float, default=0.35, help="Mapper cover overlap fraction.")
    parser.add_argument("--mapper-min-samples", type=int, default=3, help="DBSCAN min_samples in Mapper cubes.")
    parser.add_argument(
        "--mapper-eps-scale",
        type=float,
        default=1.5,
        help="Multiplier for median kNN distance used as Mapper DBSCAN eps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cmd = [
        sys.executable,
        "-m",
        RUN_METHODS_MODULE,
        "--dataset-dir",
        str(args.dataset_dir),
        "--output-dir",
        str(args.output_dir),
        "--start",
        str(args.start),
        "--methods",
        *args.methods,
        *(["--limit", str(args.limit)] if args.limit is not None else []),
        "--pdf-sample-size",
        str(args.pdf_sample_size),
        "--pdf-seed",
        str(args.pdf_seed),
        "--screebtower-k",
        str(args.screebtower_k),
        "--screebtower-filter-method",
        str(args.screebtower_filter_method),
        "--paga-knn",
        str(args.paga_knn),
        "--paga-resolution",
        str(args.paga_resolution),
        "--paga-connectivity-threshold",
        str(args.paga_connectivity_threshold),
        "--mapper-cubes",
        str(args.mapper_cubes),
        "--mapper-overlap",
        str(args.mapper_overlap),
        "--mapper-min-samples",
        str(args.mapper_min_samples),
        "--mapper-eps-scale",
        str(args.mapper_eps_scale),
    ]
    if args.force:
        cmd.append("--force")
    if args.no_pdfs:
        cmd.append("--no-pdfs")

    print("running:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
