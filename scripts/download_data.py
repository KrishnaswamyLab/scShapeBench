"""Download scShapeBench datasets from HuggingFace.

Files keep their repo paths, so with the default ``--local-dir .`` they land in
``./data/synthetic``, ``./data/scRNAseq``, ``./data/sample`` and ``./labels``.

The full scRNAseq set is ~116 GB; ``sample`` is a 9-dataset (~2.6 GB) subset.
Use ``--ids`` to fetch individual scRNAseq datasets.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO_ID = "scShape-Benchmark/scShapeBench"

SUBSETS = {
    "synthetic": ["data/synthetic/*"],
    "scRNAseq": ["data/scRNAseq/*"],
    "sample": ["data/sample/*"],
    "labels": ["labels/*"],
}


def main():
    parser = argparse.ArgumentParser(description="Download scShapeBench datasets from HuggingFace.")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("."),
        help="Directory to download into; repo paths are preserved (default: .)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=HF_REPO_ID,
        help=f"HuggingFace dataset repo ID (default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--subset",
        nargs="+",
        choices=sorted(SUBSETS),
        default=None,
        help="Which parts to download (default: synthetic, or nothing extra when --ids is given)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Only these scRNAseq dataset IDs, e.g. SCD-0010 SCD-0014 (implies the scRNAseq subset)",
    )
    args = parser.parse_args()

    subsets = args.subset if args.subset is not None else ([] if args.ids else ["synthetic"])
    patterns = [pattern for subset in subsets for pattern in SUBSETS[subset]]
    if args.ids:
        patterns += [f"data/scRNAseq/{sample_id}.h5ad" for sample_id in args.ids]

    print(f"Downloading {patterns} from {args.repo_id} → {args.local_dir}/")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.local_dir),
        allow_patterns=patterns,
    )
    print("Done.")


if __name__ == "__main__":
    main()
