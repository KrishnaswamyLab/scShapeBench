"""Download scShapeBench datasets from HuggingFace."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO_ID = "your-username/scShapeBench"


def main():
    parser = argparse.ArgumentParser(description="Download scShapeBench datasets from HuggingFace.")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("data"),
        help="Directory to download data into (default: ./data)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=HF_REPO_ID,
        help=f"HuggingFace dataset repo ID (default: {HF_REPO_ID})",
    )
    args = parser.parse_args()

    print(f"Downloading from {args.repo_id} → {args.local_dir}/")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.local_dir),
    )
    print("Done.")


if __name__ == "__main__":
    main()
