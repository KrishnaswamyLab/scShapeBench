"""Run scReebTower, smoothed scReebTower, PAGA, and Mapper baselines on synthetic data.

The runner processes one sample at a time and compares each recovered graph to
the contracted ground-truth graph. Method outputs are cached as JSON payloads,
so reruns do not regenerate expensive artifacts unless ``--force`` is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import anndata as ad
from gtda.mapper import CubicalCover, make_mapper_pipeline
import kmapper as km
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402
import scanpy as sc

BENCHMARK_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = BENCHMARK_ROOT
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

# scReebTower (formerly dReeb) is installed as the ``sc-reeb-tower`` package:
#   pip install "sc-reeb-tower @ git+https://github.com/KrishnaswamyLab/scReebTower"
try:
    from sc_reeb_tower import scReebTower as screebtower  # type: ignore  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - optional method stack
    screebtower = None
    _SCREEBTOWER_IMPORT_ERROR = exc
else:
    _SCREEBTOWER_IMPORT_ERROR = None
from ..evaluation import (  # type: ignore  # noqa: E402
    _load_truth_payload_from_npz,
    build_graph_from_payload,
    evaluate_full_graph_recovery,
    load_dataset_manifest,
)


TRACKS = ("latent",)
SCREEBTOWER_SMOOTH_METHOD = "screebtower_smooth"
SCREEBTOWER_SMOOTH_T = "auto"
SCREEBTOWER_SMOOTH_K_SMOOTH_MAX = 80

METHODS = ("screebtower", SCREEBTOWER_SMOOTH_METHOD, "paga", "mapper", "gtda_mapper")
RESULT_FIELDS = [
    "sample_id",
    "method",
    "track",
    "status",
    "topology_score",
    "persistence_score",
    "graph_edit_distance",
    "cycle_recovery_score",
    "branch_recovery_score",
    "strict_reduced_isomorphism",
    "relaxed_topology_match",
    "cycle_count_correct",
    "branch_count_correct",
    "component_count_correct",
    "component_preservation_score",
    "truth_reduced_nodes",
    "truth_reduced_edges",
    "pred_reduced_nodes",
    "pred_reduced_edges",
    "truth_betti_1",
    "pred_betti_1",
    "truth_num_components",
    "pred_num_components",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=str(BENCHMARK_ROOT / "data/synthetic"),
        help=(
            "Dataset directory containing sample_*.npz/.json (flat or under samples/). "
            "manifest.jsonl is used when present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(BENCHMARK_ROOT / "results/method_benchmark"),
        help="Directory for cached method outputs and result tables.",
    )
    parser.add_argument("--start", type=int, default=0, help="First manifest row to process.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to process.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Methods to run.",
    )
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
    parser.add_argument(
        "--pdf-sample-size",
        type=int,
        default=10,
        help="Number of random benchmark samples to include in each figure PDF.",
    )
    parser.add_argument("--pdf-seed", type=int, default=0, help="Random seed for selecting figure PDF samples.")
    parser.add_argument("--no-pdfs", action="store_true", help="Skip random-sample figure PDF generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    needs_screebtower = {"screebtower", SCREEBTOWER_SMOOTH_METHOD} & set(args.methods)
    if needs_screebtower and screebtower is None:
        raise SystemExit(
            f"{', '.join(sorted(needs_screebtower))} requested but sc-reeb-tower is not installed. "
            'Install it with: pip install "sc-reeb-tower @ git+https://github.com/KrishnaswamyLab/scReebTower" '
            "or drop it from --methods."
        )

    manifest = load_dataset_manifest(dataset_dir)
    selected = manifest[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    rows: list[dict[str, Any]] = []
    for row_index, entry in enumerate(selected, start=args.start):
        sample_id = int(entry["sample_id"])
        sample_stem = Path(entry["npz_path"]).stem
        sample_npz = dataset_dir / entry["npz_path"]
        print(f"[sample {sample_id:05d} row={row_index}] {sample_npz}", flush=True)

        sample = load_sample(sample_npz)
        truth_payload = _load_truth_payload_from_npz(sample_npz)

        for method in args.methods:
            print(f"  - {method}: point-cloud prediction", flush=True)
            try:
                pred_payload = load_or_build_payload(
                    cache_dir=cache_dir,
                    sample_stem=sample_stem,
                    method=method,
                    kind="pointcloud",
                    force=args.force,
                    builder=lambda method=method: build_pointcloud_payload(method, sample, args),
                )
            except Exception as exc:
                rows.append(error_row(sample_id, method, "latent", exc))
                write_tables(rows, output_dir)
                continue

            for track, track_truth_payload in (
                ("latent", truth_payload),
            ):
                print(f"  - {method}: evaluate {track}", flush=True)
                try:
                    eval_result = evaluate_payload_pair(
                        truth_payload=track_truth_payload,
                        pred_payload=pred_payload,
                        observed_points=sample["noisy_points"],
                    )
                    rows.append(result_row(sample_id, method, track, eval_result))
                except Exception as exc:
                    rows.append(error_row(sample_id, method, track, exc))
                write_tables(rows, output_dir)

    write_summary(rows, output_dir)
    if not args.no_pdfs:
        render_random_sample_pdfs(args, dataset_dir, output_dir, cache_dir, selected)
    print(f"wrote: {output_dir / 'results.csv'}", flush=True)
    print(f"wrote: {output_dir / 'results.md'}", flush=True)
    print(f"wrote: {output_dir / 'summary.csv'}", flush=True)
    if not args.no_pdfs:
        print(f"wrote: {output_dir / 'track_latent_random_sample.pdf'}", flush=True)


def load_sample(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "nodes": np.asarray(data["nodes"], dtype=int),
            "edges": np.asarray(data["edges"], dtype=int),
            "positions": np.asarray(data["positions"], dtype=float),
            "clean_points": np.asarray(data["clean_points"], dtype=float),
            "noisy_points": np.asarray(data["noisy_points"], dtype=float),
        }


def load_or_build_payload(
    *,
    cache_dir: Path,
    sample_stem: str,
    method: str,
    kind: str,
    force: bool,
    builder: Any,
) -> dict[str, Any]:
    path = cache_dir / method / kind / f"{sample_stem}.json"
    if path.is_file() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    payload = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return payload


def build_pointcloud_payload(
    method: str,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if method == "screebtower":
        return build_screebtower_pointcloud_payload(sample["noisy_points"], args)
    if method == SCREEBTOWER_SMOOTH_METHOD:
        return build_screebtower_smoothed_pointcloud_payload(sample["noisy_points"], args)
    if method == "paga":
        return build_paga_payload(sample["noisy_points"], args)
    if method == "mapper":
        return build_mapper_payload(sample["noisy_points"], args)
    if method == "gtda_mapper":
        return build_gtda_mapper_payload(sample["noisy_points"], args)
    raise ValueError(f"Unsupported method: {method}")


def build_screebtower_pointcloud_payload(points: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    if screebtower is None:
        raise RuntimeError(
            "Running the screebtower synthetic method requires the sc-reeb-tower package (pip install \"sc-reeb-tower @ git+https://github.com/KrishnaswamyLab/scReebTower\")."
        ) from _SCREEBTOWER_IMPORT_ERROR
    result = screebtower(
        X=points,
        k=int(args.screebtower_k),
        precision=1.0,
        filter_method=args.screebtower_filter_method,
        simplify=True,
        return_edge_lengths=True,
    )
    graph = result["simplified"]
    nodes = list(range(len(graph["nodes"])))
    edges = [
        {"u": int(u), "v": int(v), "edge_id": int(edge_id)}
        for edge_id, (u, v) in enumerate(graph["edges"])
    ]
    return {
        "graph_representation": "reduced_multigraph",
        "already_reduced_multigraph": True,
        "nodes": nodes,
        "edges": edges,
        "edge_lengths": [float(value) for value in graph.get("edge_lengths", [])],
        "method_metadata": {
            "method": "screebtower",
            "input": "pointcloud",
            "k": int(args.screebtower_k),
            "filter_method": args.screebtower_filter_method,
        },
    }


def build_screebtower_smoothed_pointcloud_payload(points: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    if screebtower is None:
        raise RuntimeError(
            "Running the screebtower_smooth synthetic method requires the sc-reeb-tower package (pip install \"sc-reeb-tower @ git+https://github.com/KrishnaswamyLab/scReebTower\")."
        ) from _SCREEBTOWER_IMPORT_ERROR
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    k_smooth = max(1, min(int(SCREEBTOWER_SMOOTH_K_SMOOTH_MAX), max(1, n - 1)))
    result = screebtower(
        X=points,
        k=int(args.screebtower_k),
        k_smooth=int(k_smooth),
        t=SCREEBTOWER_SMOOTH_T,
        precision=1.0,
        filter_method=args.screebtower_filter_method,
        simplify=True,
        return_edge_lengths=True,
    )
    graph = result["simplified"]
    nodes = list(range(len(graph["nodes"])))
    edges = [
        {"u": int(u), "v": int(v), "edge_id": int(edge_id)}
        for edge_id, (u, v) in enumerate(graph["edges"])
    ]
    return {
        "graph_representation": "reduced_multigraph",
        "already_reduced_multigraph": True,
        "nodes": nodes,
        "edges": edges,
        "edge_lengths": [float(value) for value in graph.get("edge_lengths", [])],
        "method_metadata": {
            "method": SCREEBTOWER_SMOOTH_METHOD,
            "input": "pointcloud",
            "k": int(args.screebtower_k),
            "k_smooth": int(k_smooth),
            "t": SCREEBTOWER_SMOOTH_T,
            "filter_method": args.screebtower_filter_method,
        },
    }


def build_paga_payload(points: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    if n == 0:
        return simple_payload([], [], {})

    sc.settings.verbosity = 0
    adata = ad.AnnData(points)
    knn = min(max(int(args.paga_knn), 2), n - 1)
    sc.pp.neighbors(adata, n_neighbors=knn, use_rep="X")
    sc.tl.leiden(
        adata,
        resolution=float(args.paga_resolution),
        random_state=0,
        key_added="leiden",
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    sc.tl.paga(adata, groups="leiden")

    labels_raw = np.asarray(adata.obs["leiden"].astype(str).to_numpy())
    categories = sorted(set(labels_raw), key=lambda value: int(value) if value.isdigit() else value)
    label_to_node = {label: idx for idx, label in enumerate(categories)}
    labels = np.array([label_to_node[label] for label in labels_raw], dtype=int)

    node_positions = {
        str(node_id): points[labels == node_id].mean(axis=0).tolist()
        for node_id in range(len(categories))
        if np.any(labels == node_id)
    }

    connectivities = adata.uns["paga"]["connectivities"].tocoo()
    edges = set()
    threshold = float(args.paga_connectivity_threshold)
    for u, v, value in zip(connectivities.row, connectivities.col, connectivities.data):
        u = int(u)
        v = int(v)
        if u == v or float(value) <= threshold:
            continue
        edges.add((u, v) if u < v else (v, u))

    return {
        "graph_representation": "simple_graph",
        "nodes": list(range(len(categories))),
        "edges": [list(edge) for edge in sorted(edges)],
        "node_positions": node_positions,
        "method_metadata": {
            "method": "scanpy_paga",
            "clusters": int(len(categories)),
            "knn": int(knn),
            "leiden_resolution": float(args.paga_resolution),
            "connectivity_threshold": threshold,
        },
    }


def build_mapper_payload(points: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    if n == 0:
        return simple_payload([], [], {})

    eps = mapper_eps(points, min_samples=max(2, int(args.mapper_min_samples)), scale=float(args.mapper_eps_scale))
    mapper = km.KeplerMapper(verbose=0)
    lens_dim = min(2, points.shape[1], n)
    projection: Any
    if lens_dim >= 1:
        projection = PCA(n_components=lens_dim, random_state=0)
    else:
        projection = "sum"
    lens = mapper.fit_transform(points, projection=projection)
    graph = mapper.map(
        lens,
        X=points,
        clusterer=DBSCAN(eps=eps, min_samples=int(args.mapper_min_samples)),
        cover=km.Cover(n_cubes=int(args.mapper_cubes), perc_overlap=float(args.mapper_overlap)),
        remove_duplicate_nodes=True,
    )

    node_keys = sorted(graph.get("nodes", {}).keys())
    key_to_node = {key: idx for idx, key in enumerate(node_keys)}
    memberships = [
        np.asarray(graph["nodes"][key], dtype=int)
        for key in node_keys
    ]

    if not memberships:
        centroid = points.mean(axis=0)
        return {
            "graph_representation": "simple_graph",
            "nodes": [0],
            "edges": [],
            "node_positions": {"0": centroid.tolist()},
            "method_metadata": {
                "method": "mapper",
                "fallback": "single_node",
                "eps": float(eps),
            },
        }

    node_positions = {
        str(node_id): points[members].mean(axis=0).tolist()
        for node_id, members in enumerate(memberships)
    }
    edges = set()
    for source_key, target_keys in graph.get("links", {}).items():
        if source_key not in key_to_node:
            continue
        u = key_to_node[source_key]
        for target_key in target_keys:
            if target_key not in key_to_node:
                continue
            v = key_to_node[target_key]
            if u == v:
                continue
            edges.add((u, v) if u < v else (v, u))

    return {
        "graph_representation": "simple_graph",
        "nodes": list(range(len(memberships))),
        "edges": [list(edge) for edge in sorted(edges)],
        "node_positions": node_positions,
        "method_metadata": {
            "method": "mapper",
            "package": "kepler-mapper",
            "n_cubes": int(args.mapper_cubes),
            "overlap": float(args.mapper_overlap),
            "min_samples": int(args.mapper_min_samples),
            "eps": float(eps),
            "num_cover_nodes": len(memberships),
        },
    }


def build_gtda_mapper_payload(points: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    if n == 0:
        return simple_payload([], [], {})

    eps = mapper_eps(points, min_samples=max(2, int(args.mapper_min_samples)), scale=float(args.mapper_eps_scale))
    lens_dim = min(2, points.shape[1], n)
    filter_func: Any = PCA(n_components=lens_dim, random_state=0) if lens_dim >= 1 else None
    pipeline = make_mapper_pipeline(
        filter_func=filter_func,
        cover=CubicalCover(
            n_intervals=int(args.mapper_cubes),
            overlap_frac=float(args.mapper_overlap),
        ),
        clusterer=DBSCAN(eps=eps, min_samples=int(args.mapper_min_samples)),
        graph_step=True,
        store_edge_elements=False,
        contract_nodes=True,
        verbose=False,
    )
    graph = pipeline.fit_transform(points)
    memberships = [
        np.asarray(graph.vs[node_id]["node_elements"], dtype=int)
        for node_id in range(graph.vcount())
    ]

    if not memberships:
        centroid = points.mean(axis=0)
        return {
            "graph_representation": "simple_graph",
            "nodes": [0],
            "edges": [],
            "node_positions": {"0": centroid.tolist()},
            "method_metadata": {
                "method": "gtda_mapper",
                "fallback": "single_node",
                "eps": float(eps),
            },
        }

    node_positions = {
        str(node_id): points[members].mean(axis=0).tolist()
        for node_id, members in enumerate(memberships)
    }
    edges = sorted(
        {
            (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
            for u, v in graph.get_edgelist()
            if int(u) != int(v)
        }
    )
    return {
        "graph_representation": "simple_graph",
        "nodes": list(range(len(memberships))),
        "edges": [list(edge) for edge in edges],
        "node_positions": node_positions,
        "method_metadata": {
            "method": "gtda_mapper",
            "package": "giotto-tda",
            "n_intervals": int(args.mapper_cubes),
            "overlap_frac": float(args.mapper_overlap),
            "min_samples": int(args.mapper_min_samples),
            "eps": float(eps),
            "num_cover_nodes": len(memberships),
        },
    }


def mapper_eps(points: np.ndarray, min_samples: int, scale: float) -> float:
    n = points.shape[0]
    if n <= 1:
        return 1.0
    k = min(max(min_samples, 2), n)
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    base = float(np.median(distances[:, -1]))
    if not math.isfinite(base) or base <= 0.0:
        pairwise = pdist(points)
        positive = pairwise[pairwise > 0.0]
        base = float(np.median(positive)) if positive.size else 1.0
    return max(float(scale) * base, np.finfo(float).eps)


def payload_from_multigraph(graph: nx.MultiGraph) -> dict[str, Any]:
    nodes = list(graph.nodes())
    edges = []
    edge_lengths = []
    for edge_id, (u, v, key, data) in enumerate(graph.edges(keys=True, data=True)):
        edges.append({"u": int(u), "v": int(v), "edge_id": int(key) if isinstance(key, int) else int(edge_id)})
        edge_lengths.append(float(data.get("total_length", data.get("length", 1.0))))
    return {
        "graph_representation": "reduced_multigraph",
        "already_reduced_multigraph": True,
        "nodes": [int(node) for node in nodes],
        "edges": edges,
        "edge_lengths": edge_lengths,
        "method_metadata": {
            "method": "screebtower",
            "input": "latent_graph",
        },
    }


def simple_payload(nodes: list[int], edges: list[list[int]], positions: dict[int, Any]) -> dict[str, Any]:
    return {
        "graph_representation": "simple_graph",
        "nodes": nodes,
        "edges": edges,
        "node_positions": {str(node): value for node, value in positions.items()},
    }


def evaluate_payload_pair(
    *,
    truth_payload: dict[str, Any],
    pred_payload: dict[str, Any],
    observed_points: np.ndarray,
) -> dict[str, Any]:
    truth_graph, truth_positions, truth_already_reduced = build_graph_from_payload(
        truth_payload,
        positions_required_for_persistence=True,
    )
    pred_graph, pred_positions, pred_already_reduced = build_graph_from_payload(
        pred_payload,
        positions_required_for_persistence=True,
    )
    return evaluate_full_graph_recovery(
        truth_graph=truth_graph,
        pred_graph=pred_graph,
        truth_positions=truth_positions,
        pred_positions=pred_positions,
        observed_points=observed_points,
        truth_already_reduced_multigraph=truth_already_reduced,
        pred_already_reduced_multigraph=pred_already_reduced,
    )


def result_row(sample_id: int, method: str, track: str, result: dict[str, Any]) -> dict[str, Any]:
    topology = result["topology"]
    relaxed = result["relaxed_topology"]
    count = result["count_recovery"]
    truth_summary = relaxed["truth_summary"]
    pred_summary = relaxed["pred_summary"]
    return {
        "sample_id": sample_id,
        "method": method,
        "track": track,
        "status": "ok",
        "topology_score": result["headline_scores"]["topology_score"],
        "persistence_score": result["headline_scores"]["persistence_score"],
        "graph_edit_distance": result["graph_edit_distance"],
        "cycle_recovery_score": result["headline_scores"]["cycle_recovery_score"],
        "branch_recovery_score": result["headline_scores"]["branch_recovery_score"],
        "strict_reduced_isomorphism": topology["exact_match_up_to_degree2_suppression"],
        "relaxed_topology_match": relaxed["relaxed_topology_match"],
        "cycle_count_correct": count["cycle_count_correct"],
        "branch_count_correct": count["branch_count_correct"],
        "component_count_correct": count["component_count_correct"],
        "component_preservation_score": count["component_preservation_score"],
        "truth_reduced_nodes": truth_summary["num_nodes"],
        "truth_reduced_edges": truth_summary["num_edges"],
        "pred_reduced_nodes": pred_summary["num_nodes"],
        "pred_reduced_edges": pred_summary["num_edges"],
        "truth_betti_1": truth_summary["betti_1"],
        "pred_betti_1": pred_summary["betti_1"],
        "truth_num_components": count["truth_num_components"],
        "pred_num_components": count["pred_num_components"],
        "error": "",
    }


def error_row(sample_id: int, method: str, track: str, exc: Exception) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "method": method,
        "track": track,
        "status": "error",
        "topology_score": "",
        "persistence_score": "",
        "graph_edit_distance": "",
        "cycle_recovery_score": "",
        "branch_recovery_score": "",
        "strict_reduced_isomorphism": "",
        "relaxed_topology_match": "",
        "cycle_count_correct": "",
        "branch_count_correct": "",
        "component_count_correct": "",
        "component_preservation_score": "",
        "truth_reduced_nodes": "",
        "truth_reduced_edges": "",
        "pred_reduced_nodes": "",
        "pred_reduced_edges": "",
        "truth_betti_1": "",
        "pred_betti_1": "",
        "truth_num_components": "",
        "pred_num_components": "",
        "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
    }


def write_tables(rows: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = output_dir / "results.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(RESULT_FIELDS) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(RESULT_FIELDS)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(format_markdown_cell(row.get(field, "")) for field in RESULT_FIELDS) + " |\n")


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["track"])), []).append(row)

    summary_fields = [
        "method",
        "track",
        "num_rows",
        "num_ok",
        "num_error",
        "mean_topology_score",
        "mean_persistence_score",
        "mean_graph_edit_distance",
        "mean_cycle_recovery_score",
        "mean_branch_recovery_score",
        "strict_match_rate",
        "relaxed_match_rate",
        "cycle_count_recovery_rate",
        "branch_count_recovery_rate",
        "component_count_recovery_rate",
        "mean_component_preservation_score",
        "graph_edit_distance_exact_rate",
    ]
    summary_rows = []
    for (method, track), group_rows in sorted(groups.items()):
        ok_rows = [row for row in group_rows if row["status"] == "ok"]
        summary_rows.append(
            {
                "method": method,
                "track": track,
                "num_rows": len(group_rows),
                "num_ok": len(ok_rows),
                "num_error": len(group_rows) - len(ok_rows),
                "mean_topology_score": mean_field(ok_rows, "topology_score"),
                "mean_persistence_score": mean_field(ok_rows, "persistence_score"),
                "mean_graph_edit_distance": mean_field(ok_rows, "graph_edit_distance"),
                "mean_cycle_recovery_score": mean_field(ok_rows, "cycle_recovery_score"),
                "mean_branch_recovery_score": mean_field(ok_rows, "branch_recovery_score"),
                "strict_match_rate": mean_bool(ok_rows, "strict_reduced_isomorphism"),
                "relaxed_match_rate": mean_bool(ok_rows, "relaxed_topology_match"),
                "cycle_count_recovery_rate": mean_bool(ok_rows, "cycle_count_correct"),
                "branch_count_recovery_rate": mean_bool(ok_rows, "branch_count_correct"),
                "component_count_recovery_rate": mean_bool(ok_rows, "component_count_correct"),
                "mean_component_preservation_score": mean_field(ok_rows, "component_preservation_score"),
                "graph_edit_distance_exact_rate": mean_bool(ok_rows, "graph_edit_distance_exact"),
            }
        )

    path = output_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    write_summary_latex(rows, output_dir)


def write_summary_latex(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["track"])), []).append(row)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Synthetic benchmark summary on the full recovered-graph track. "
        r"Graph edit distance (GED) is reported as a lower-is-better score; "
        r"persistence similarity is reported as a higher-is-better score. "
        r"Values are mean $\pm$ std over benchmark samples.}"
    )
    lines.append(r"\label{tab:synthetic-summary}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Track & GED $\downarrow$ & Persistence similarity $\uparrow$ \\")
    lines.append(r"\midrule")

    for (method, track), group_rows in sorted(groups.items()):
        ok_rows = [row for row in group_rows if row["status"] == "ok"]
        ged_mean, ged_std = field_mean_std(ok_rows, "graph_edit_distance")
        pers_mean, pers_std = field_mean_std(ok_rows, "persistence_score")
        lines.append(
            f"{latex_escape(method)} & {latex_escape(track)} & "
            f"{fmt_pm(ged_mean, ged_std)} & {fmt_pm(pers_mean, pers_std)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = output_dir / "summary.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote: {path}", flush=True)


def render_random_sample_pdfs(
    args: argparse.Namespace,
    dataset_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    selected_entries: list[dict[str, Any]],
) -> None:
    sample_size = max(0, int(args.pdf_sample_size))
    if sample_size == 0 or not selected_entries:
        return

    rng = np.random.default_rng(int(args.pdf_seed))
    count = min(sample_size, len(selected_entries))
    chosen_indices = sorted(rng.choice(len(selected_entries), size=count, replace=False).tolist())
    chosen_entries = [selected_entries[index] for index in chosen_indices]

    for track in TRACKS:
        pdf_path = output_dir / f"track_{track}_random_sample.pdf"
        with PdfPages(pdf_path) as pdf:
            for entry in chosen_entries:
                sample_id = int(entry["sample_id"])
                sample_stem = Path(entry["npz_path"]).stem
                sample_npz = dataset_dir / entry["npz_path"]
                sample = load_sample(sample_npz)
                truth_payload = _load_truth_payload_from_npz(sample_npz)
                panels: list[tuple[str, str, Any]] = []
                if track == "latent":
                    panels.append(("graph_original", "Target: contracted original graph", sample))
                panels.append(("point_cloud", "Observed noisy point cloud", sample["noisy_points"]))

                for method in args.methods:
                    payload, error = load_cached_payload_optional(cache_dir, sample_stem, method, "pointcloud")
                    if payload is None:
                        panels.append(("message", method_label(method), error))
                    else:
                        panels.append(("payload_graph", method_label(method), payload))

                fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
                fig.suptitle(f"sample {sample_id:05d} - {track_label(track)}", fontsize=15, fontweight="bold")
                for ax, panel in zip(axes.ravel(), panels):
                    panel_kind, title, value = panel
                    if panel_kind == "graph_original":
                        draw_original_graph(ax, value, title)
                    elif panel_kind == "point_cloud":
                        draw_point_cloud(ax, value, title)
                    elif panel_kind == "message":
                        draw_message(ax, title, str(value))
                    else:
                        draw_payload_graph(ax, value, title)
                for ax in axes.ravel()[len(panels) :]:
                    ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)


def load_cached_payload(cache_dir: Path, sample_stem: str, method: str, kind: str) -> dict[str, Any]:
    path = cache_dir / method / kind / f"{sample_stem}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached payload for PDF rendering: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_cached_payload_optional(
    cache_dir: Path,
    sample_stem: str,
    method: str,
    kind: str,
) -> tuple[dict[str, Any] | None, str]:
    try:
        return load_cached_payload(cache_dir, sample_stem, method, kind), ""
    except Exception as exc:
        return None, "".join(traceback.format_exception_only(type(exc), exc)).strip()


def method_label(method: str) -> str:
    labels = {
        "screebtower": "scReebTower(X)",
        "paga": "Scanpy PAGA(X)",
        "mapper": "KeplerMapper(X)",
        "gtda_mapper": "giotto-tda Mapper(X)",
    }
    return labels.get(method, method)


def track_label(track: str) -> str:
    if track == "latent":
        return "Point-cloud output vs contracted original graph"
    return track


def draw_original_graph(ax: Any, sample: dict[str, Any], title: str) -> None:
    graph = nx.Graph()
    nodes = [int(node) for node in sample["nodes"]]
    graph.add_nodes_from(nodes)
    for u, v in np.asarray(sample["edges"], dtype=int):
        graph.add_edge(int(u), int(v))
    positions_2d = project_points_2d(np.asarray(sample["positions"], dtype=float))
    pos = {node: positions_2d[index] for index, node in enumerate(nodes)}
    draw_graph_with_layout(ax, graph, pos, title)


def draw_point_cloud(ax: Any, points: np.ndarray, title: str) -> None:
    points_2d = project_points_2d(np.asarray(points, dtype=float))
    ax.scatter(points_2d[:, 0], points_2d[:, 1], s=4, alpha=0.55, c="#2f6f9f", linewidths=0)
    finish_axis(ax, f"{title}\nn={len(points_2d)}")


def draw_payload_graph(ax: Any, payload: dict[str, Any], title: str) -> None:
    graph, positions, _ = build_graph_from_payload(payload, positions_required_for_persistence=False)
    if graph.number_of_nodes() == 0:
        finish_axis(ax, f"{title}\nempty graph")
        return

    pos = positions_to_2d(graph, positions)
    draw_graph_with_layout(ax, graph, pos, title)


def draw_message(ax: Any, title: str, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message or "missing",
        ha="center",
        va="center",
        fontsize=9,
        color="#7a1f1f",
        wrap=True,
        transform=ax.transAxes,
    )
    finish_axis(ax, title)


def positions_to_2d(graph: nx.Graph, positions: dict[Any, Any] | None) -> dict[Any, np.ndarray]:
    if positions:
        nodes = list(graph.nodes())
        available_nodes = [node for node in nodes if node in positions]
        if len(available_nodes) == graph.number_of_nodes():
            points = np.asarray([positions[node] for node in nodes], dtype=float)
            points_2d = project_points_2d(points)
            return {node: points_2d[index] for index, node in enumerate(nodes)}
    return {
        node: np.asarray(coord, dtype=float)
        for node, coord in nx.spring_layout(graph, seed=0, weight="length").items()
    }


def draw_graph_with_layout(ax: Any, graph: nx.Graph, pos: dict[Any, np.ndarray], title: str) -> None:
    draw_edges(ax, graph, pos)
    if graph.number_of_nodes():
        xy = np.asarray([pos[node] for node in graph.nodes()], dtype=float)
        ax.scatter(xy[:, 0], xy[:, 1], s=42, c="#f4f7fb", edgecolors="#1c2938", linewidths=0.9, zorder=3)
    label = f"{title}\n|V|={graph.number_of_nodes()} |E|={graph.number_of_edges()} beta1={betti_1(graph)}"
    finish_axis(ax, label)


def draw_edges(ax: Any, graph: nx.Graph, pos: dict[Any, np.ndarray]) -> None:
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        edge_items = list(graph.edges(keys=True))
    else:
        edge_items = [(u, v, 0) for u, v in graph.edges()]
    grouped: dict[tuple[Any, Any], list[tuple[Any, Any, Any]]] = {}
    for u, v, key in edge_items:
        group_key = (u, v) if str(u) <= str(v) else (v, u)
        grouped.setdefault(group_key, []).append((u, v, key))

    for edges in grouped.values():
        total = len(edges)
        for index, (u, v, _key) in enumerate(edges):
            if u == v:
                draw_self_loop(ax, pos[u], index, total)
                continue
            rad = 0.0
            if total > 1:
                offset = index - (total - 1) / 2.0
                rad = 0.18 * offset
                if abs(rad) < 0.03:
                    rad = 0.03
            patch = FancyArrowPatch(
                posA=pos[u],
                posB=pos[v],
                arrowstyle="-",
                connectionstyle=f"arc3,rad={rad}",
                color="#44546a",
                linewidth=1.2,
                alpha=0.82,
                zorder=1,
            )
            ax.add_patch(patch)


def draw_self_loop(ax: Any, center: np.ndarray, index: int, total: int) -> None:
    radius = 0.045 + 0.018 * index
    if total > 1:
        radius += 0.012 * total
    circle = Circle(
        (float(center[0]) + radius, float(center[1]) + radius),
        radius,
        fill=False,
        edgecolor="#44546a",
        linewidth=1.1,
        alpha=0.82,
        zorder=1,
    )
    ax.add_patch(circle)


def project_points_2d(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros((0, 2), dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    if points.shape[1] >= 2:
        projected = points[:, :2]
    elif points.shape[0] >= 2:
        projected = np.column_stack([points[:, 0], np.zeros(points.shape[0])])
    else:
        projected = np.zeros((points.shape[0], 2), dtype=float)
    if points.shape[1] > 2 and points.shape[0] >= 3:
        projected = PCA(n_components=2, random_state=0).fit_transform(points)
    return normalize_points(projected)


def normalize_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    points = np.asarray(points, dtype=float)
    finite = np.isfinite(points).all(axis=1)
    if not np.all(finite):
        points = points.copy()
        points[~finite] = 0.0
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, np.finfo(float).eps)
    return (points - mins) / span


def betti_1(graph: nx.Graph) -> int:
    return int(graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph))


def finish_axis(ax: Any, title: str) -> None:
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.15)
    for spine in ax.spines.values():
        spine.set_visible(False)


def mean_field(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
    return float(sum(values) / len(values)) if values else ""


def mean_bool(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [bool(row[field]) for row in rows if row.get(field) not in ("", None)]
    return float(sum(1.0 for value in values if value) / len(values)) if values else ""


def field_mean_std(rows: list[dict[str, Any]], field: str) -> tuple[float, float]:
    values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def fmt_pm(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return r"\text{---}"
    if not np.isfinite(std):
        return f"{mean:.3f}"
    return f"{mean:.3f}$\\pm${std:.3f}"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def format_markdown_cell(value: Any) -> str:
    text = str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    if len(text) > 160:
        text = text[:157] + "..."
    return text


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
