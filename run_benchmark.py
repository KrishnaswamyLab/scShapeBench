"""Unified benchmark dispatcher for real and synthetic tracks."""

from __future__ import annotations

import argparse
import sys


def _dispatch_real(mode: str, argv: list[str]) -> None:
    if mode == "all":
        from run_benchmark_all import main as entrypoint
    elif mode == "single":
        from run_benchmark_single_cell import main as entrypoint
    else:
        raise SystemExit(f"Unknown real benchmark mode: {mode!r}")
    sys.argv = [sys.argv[0], *argv]
    entrypoint()


def _dispatch_synthetic(argv: list[str]) -> None:
    from run_benchmark_synthetic import main as entrypoint
    sys.argv = [sys.argv[0], *argv]
    entrypoint()


def _build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, add_help=add_help)
    if not add_help:
        parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit.")
    parser.add_argument("--track", choices=["real", "synthetic"], required=True)
    parser.add_argument(
        "--real-mode",
        choices=["all", "single"],
        default="all",
        help="Real-track entrypoint to use.",
    )
    return parser


def main() -> None:
    argv = sys.argv[1:]
    parser = _build_parser(add_help=False)
    if not argv or argv == ["--help"] or argv == ["-h"]:
        _build_parser().parse_args(["--help"])
    args, remainder = parser.parse_known_args(argv)

    if args.help:
        remainder.append("--help")

    if args.track == "real":
        _dispatch_real(args.real_mode, remainder)
    else:
        _dispatch_synthetic(remainder)


if __name__ == "__main__":
    main()
