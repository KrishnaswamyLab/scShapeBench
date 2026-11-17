"""Re-package graphs from another pipeline as a real-track submission.

Takes an existing method folder (``<src>/SCD-XXXX/{nodes.csv,edges.csv}``),
keeps only the graph itself, and writes the four files the scorer reads::

    <dst>/SCD-XXXX/nodes.csv        node,x,y               (x, y passed through)
    <dst>/SCD-XXXX/edges.csv        source,target,length   (extra columns dropped)
    <dst>/SCD-XXXX/h0_finite.csv    recomputed from the edge lengths
    <dst>/SCD-XXXX/h1_essential.csv recomputed from the edge lengths

Any persistence files in ``<src>`` are ignored, so imported graphs get exactly the
same persistence pipeline as graphs built by ``build_real_submissions.py`` or
``examples/knn_submission.py``. ``source``/``target`` must index rows of
``nodes.csv`` (0-based), as the scorer expects; an empty ``length`` is read as 0,
as the scorer's loader does.

    python scripts/import_graphs.py --src other_pipeline/dreeb_k25 --dst submissions/scRNAseq/dreeb_k25
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_real_submissions import write_persistence


def import_sample(src: Path, dst: Path) -> tuple[int, int, int]:
    with open(src / "nodes.csv") as handle:
        nodes = [(float(row["x"]), float(row["y"])) for row in csv.DictReader(handle)]
    edges, n_empty = [], 0
    with open(src / "edges.csv") as handle:
        for row in csv.DictReader(handle):
            u, v = int(row["source"]), int(row["target"])
            if not (0 <= u < len(nodes) and 0 <= v < len(nodes)):
                raise ValueError(f"{src}: edge ({u}, {v}) indexes past {len(nodes)} nodes")
            if row.get("length") in (None, ""):
                n_empty += 1
            edges.append((u, v, float(row["length"]) if row.get("length") else 0.0))

    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "nodes.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node", "x", "y"])
        writer.writerows([i, x, y] for i, (x, y) in enumerate(nodes))
    with open(dst / "edges.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "length"])
        writer.writerows(edges)
    write_persistence(dst, list(range(len(nodes))), edges)
    return len(nodes), len(edges), n_empty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Existing method folder holding SCD-XXXX/ subfolders.")
    parser.add_argument("--dst", type=Path, required=True, help="Method folder to write.")
    parser.add_argument("--ids", nargs="+", default=None, help="Only these dataset IDs (default: all).")
    args = parser.parse_args()

    samples = sorted(d for d in args.src.iterdir() if (d / "edges.csv").is_file())
    if args.ids:
        samples = [d for d in samples if d.name in set(args.ids)]
    if not samples:
        raise SystemExit(f"No SCD-XXXX/edges.csv under {args.src}")
    for sample in samples:
        n_nodes, n_edges, n_empty = import_sample(sample, args.dst / sample.name)
        note = f" ({n_empty} empty lengths read as 0)" if n_empty else ""
        print(f"{sample.name}: {n_nodes} nodes, {n_edges} edges{note}", flush=True)


if __name__ == "__main__":
    main()
