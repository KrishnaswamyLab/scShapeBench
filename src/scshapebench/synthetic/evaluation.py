"""Evaluation utilities for graph-recovery benchmark tasks.

This module implements the current benchmark scoring stack:

1. topology recovery up to degree-2 suppression
2. cycle-core recovery on the reduced 2-core
3. branch-forest recovery on the reduced non-cycle attachments
4. persistence similarity on reduced weighted graphs

It also provides dataset-level aggregation over a benchmark manifest,
including axis-binned reporting over noise, crowding, and density.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

CURRENT_PYTHON_TAG = f"python{sys.version_info.major}.{sys.version_info.minor}"


def _is_mismatched_site_packages(entry: str) -> bool:
    return "site-packages" in entry and "python3." in entry and CURRENT_PYTHON_TAG not in entry


def sanitize_python_environment() -> None:
    sys.path[:] = [entry for entry in sys.path if not _is_mismatched_site_packages(entry)]
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        kept_entries = [
            entry
            for entry in pythonpath.split(os.pathsep)
            if entry and not _is_mismatched_site_packages(entry)
        ]
        if kept_entries:
            os.environ["PYTHONPATH"] = os.pathsep.join(kept_entries)
        else:
            os.environ.pop("PYTHONPATH", None)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")


sanitize_python_environment()

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .graph_primitives import betti_1, branch_nodes, leaf_count

__all__ = [
    "build_graph_from_payload",
    "compare_branch_forest",
    "compare_cycle_core",
    "compare_graph_topology",
    "compare_persistence_diagrams",
    "compute_edge_length_persistence_diagrams",
    "evaluate_dataset_predictions",
    "evaluate_full_graph_recovery",
    "reduce_graph_topology",
]


def build_graph_from_payload(
    payload: dict[str, Any],
    *,
    positions_required_for_persistence: bool = False,
) -> tuple[nx.Graph | nx.MultiGraph, dict[int, np.ndarray] | None, bool]:
    """Construct a graph and optional node-position map from a payload."""

    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list):
        raise ValueError("Payload must contain a 'nodes' list.")
    if not isinstance(edges, list):
        raise ValueError("Payload must contain an 'edges' list.")

    graph_representation = payload.get("graph_representation")
    if graph_representation is not None:
        if graph_representation not in {"simple_graph", "reduced_multigraph"}:
            raise ValueError(
                "graph_representation must be 'simple_graph' or 'reduced_multigraph'."
            )
        already_reduced_multigraph = graph_representation == "reduced_multigraph"
    else:
        already_reduced_multigraph = bool(payload.get("already_reduced_multigraph", False))
    graph: nx.Graph | nx.MultiGraph
    graph = nx.MultiGraph() if already_reduced_multigraph else nx.Graph()
    graph.add_nodes_from(nodes)
    edge_records: list[tuple[Any, Any, Any | None]] = []
    for edge_index, edge in enumerate(edges):
        if isinstance(edge, dict):
            if "u" not in edge or "v" not in edge:
                raise ValueError("Each edge dict must contain 'u' and 'v' entries.")
            u = edge["u"]
            v = edge["v"]
            edge_key = edge.get("edge_id", edge_index)
        elif isinstance(edge, (list, tuple)) and len(edge) == 2:
            u, v = edge
            edge_key = edge_index if already_reduced_multigraph else None
        else:
            raise ValueError("Each edge must be a length-2 list/tuple or a dict with 'u' and 'v'.")
        if u not in graph or v not in graph:
            raise ValueError(f"Edge {(u, v)!r} references a node missing from 'nodes'.")
        if already_reduced_multigraph:
            graph.add_edge(u, v, key=edge_key)
        else:
            graph.add_edge(u, v)
        edge_records.append((u, v, edge_key))

    positions = _parse_positions(payload, nodes)
    edge_lengths = payload.get("edge_lengths")
    if edge_lengths is not None:
        _attach_edge_lengths(graph, edge_records, edge_lengths)

    if positions is not None:
        if isinstance(graph, nx.MultiGraph):
            for u, v, key in graph.edges(keys=True):
                if "length" not in graph[u][v][key] and "total_length" not in graph[u][v][key]:
                    graph[u][v][key]["length"] = float(np.linalg.norm(positions[u] - positions[v]))
        else:
            for u, v in graph.edges():
                if "length" not in graph[u][v] and "total_length" not in graph[u][v]:
                    graph[u][v]["length"] = float(np.linalg.norm(positions[u] - positions[v]))

    if positions_required_for_persistence and not _graph_has_edge_lengths(graph):
        raise ValueError(
            "Persistence evaluation requires either node_positions or edge_lengths in the payload."
        )
    return graph, positions, already_reduced_multigraph


def reduce_graph_topology(
    G: nx.Graph,
    *,
    positions: dict[int, np.ndarray] | None = None,
) -> nx.MultiGraph:
    """Suppress degree-2 chains and return a weighted multigraph skeleton."""

    if G.is_directed():
        raise ValueError("Only undirected graphs are supported.")

    reduced = nx.MultiGraph()
    if G.number_of_nodes() == 0:
        return reduced

    positions = positions or {}
    edge_lengths = _resolve_edge_lengths(G, positions)
    visited_edges: set[tuple[Any, Any]] = set()
    next_reduced_node = 0

    for component_nodes in nx.connected_components(G):
        component = G.subgraph(component_nodes)
        critical_nodes = [node for node in component.nodes() if component.degree(node) != 2]

        if not critical_nodes:
            cycle_node = next_reduced_node
            next_reduced_node += 1
            cycle_length = sum(edge_lengths[_edge_key(u, v)] for u, v in component.edges())
            reduced.add_node(cycle_node, kind="pure_cycle_component")
            reduced.add_edge(
                cycle_node,
                cycle_node,
                path_nodes=sorted(component.nodes()),
                path_edges=[list(edge) for edge in sorted(_edge_key(u, v) for u, v in component.edges())],
                suppressed_node_count=max(component.number_of_nodes() - 1, 0),
                total_length=float(cycle_length),
            )
            continue

        node_map = {node: next_reduced_node + idx for idx, node in enumerate(sorted(critical_nodes))}
        next_reduced_node += len(node_map)
        for original_node, reduced_node in node_map.items():
            reduced.add_node(reduced_node, original_node=original_node, kind="critical")

        for start in critical_nodes:
            for neighbor in component.neighbors(start):
                initial_edge = _edge_key(start, neighbor)
                if initial_edge in visited_edges:
                    continue
                path_nodes = [start]
                path_edges: list[tuple[Any, Any]] = []
                total_length = 0.0
                previous = start
                current = neighbor

                while True:
                    edge = _edge_key(previous, current)
                    visited_edges.add(edge)
                    path_edges.append(edge)
                    total_length += edge_lengths[edge]
                    path_nodes.append(current)

                    if component.degree(current) != 2:
                        reduced.add_edge(
                            node_map[start],
                            node_map[current],
                            path_nodes=list(path_nodes),
                            path_edges=[list(item) for item in path_edges],
                            suppressed_node_count=max(len(path_nodes) - 2, 0),
                            total_length=float(total_length),
                        )
                        break

                    nxt = next(node for node in component.neighbors(current) if node != previous)
                    previous, current = current, nxt

    return reduced


def compare_graph_topology(
    truth_graph: nx.Graph,
    pred_graph: nx.Graph,
    *,
    truth_positions: dict[int, np.ndarray] | None = None,
    pred_positions: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare recovered and ground-truth graphs with subdivision invariance."""

    truth_reduced = reduce_graph_topology(truth_graph, positions=truth_positions)
    pred_reduced = reduce_graph_topology(pred_graph, positions=pred_positions)
    return _compare_graph_topology_on_reduced(
        truth_graph=truth_graph,
        pred_graph=pred_graph,
        truth_reduced=truth_reduced,
        pred_reduced=pred_reduced,
    )


def compare_cycle_core(
    truth_graph: nx.Graph,
    pred_graph: nx.Graph,
    *,
    truth_positions: dict[int, np.ndarray] | None = None,
    pred_positions: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare predicted and true cycle cores on reduced graphs."""

    truth_reduced = reduce_graph_topology(truth_graph, positions=truth_positions)
    pred_reduced = reduce_graph_topology(pred_graph, positions=pred_positions)
    return _compare_cycle_core_on_reduced(truth_reduced, pred_reduced)


def compare_branch_forest(
    truth_graph: nx.Graph,
    pred_graph: nx.Graph,
    *,
    truth_positions: dict[int, np.ndarray] | None = None,
    pred_positions: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare predicted and true branch forests on reduced graphs."""

    truth_reduced = reduce_graph_topology(truth_graph, positions=truth_positions)
    pred_reduced = reduce_graph_topology(pred_graph, positions=pred_positions)
    return _compare_branch_forest_on_reduced(truth_reduced, pred_reduced)


def compute_edge_length_persistence_diagrams(
    G: nx.Graph | nx.MultiGraph,
    *,
    positions: dict[int, np.ndarray] | None = None,
    normalize_edge_lengths: bool = True,
    already_reduced_multigraph: bool = False,
) -> dict[str, Any]:
    """Extract simple weighted-graph persistence diagrams from a reduced graph.

    The graph filtration adds edges in nondecreasing length order:

    - H0 finite points are (0, death_length) for merges in Kruskal order
    - H1 essential points are (birth_length, inf) for non-tree edges

    This is a graph filtration, so H1 classes are essential rather than finite.
    """

    reduced = normalize_graph_for_evaluation(
        G,
        positions=positions,
        already_reduced_multigraph=already_reduced_multigraph,
    )
    return compute_reduced_multigraph_persistence_diagrams(
        reduced,
        normalize_edge_lengths=normalize_edge_lengths,
    )


def compute_reduced_multigraph_persistence_diagrams(
    reduced: nx.MultiGraph,
    *,
    normalize_edge_lengths: bool = True,
) -> dict[str, Any]:
    """Extract weighted persistence summaries directly from a reduced multigraph."""

    edge_lengths = [
        float(data.get("total_length", data.get("length", 1.0)))
        for _, _, _, data in reduced.edges(keys=True, data=True)
    ]
    normalized_lengths, length_scale = _normalize_lengths(
        edge_lengths,
        normalize_edge_lengths=normalize_edge_lengths,
    )

    weighted_edges: list[tuple[float, Any, Any, int]] = []
    for idx, (u, v, key, data) in enumerate(reduced.edges(keys=True, data=True)):
        weighted_edges.append((normalized_lengths[idx], u, v, key))
        data["normalized_total_length"] = normalized_lengths[idx]

    weighted_edges.sort(key=lambda item: (item[0], min(item[1], item[2]), max(item[1], item[2]), item[3]))

    union_find = nx.utils.UnionFind(reduced.nodes())
    h0_finite: list[list[float]] = []
    h1_essential: list[list[float | str]] = []

    for weight, u, v, _ in weighted_edges:
        if union_find[u] != union_find[v]:
            union_find.union(u, v)
            h0_finite.append([0.0, float(weight)])
        else:
            h1_essential.append([float(weight), "inf"])

    return {
        "edge_length_scale": float(length_scale),
        "normalize_edge_lengths": bool(normalize_edge_lengths),
        "reduced_graph_num_nodes": int(reduced.number_of_nodes()),
        "reduced_graph_num_edges": int(reduced.number_of_edges()),
        "h0_finite": h0_finite,
        "h0_essential_count": int(nx.number_connected_components(reduced)),
        "h1_essential": h1_essential,
    }


def compare_persistence_diagrams(
    truth_diagrams: dict[str, Any],
    pred_diagrams: dict[str, Any],
    *,
    persistence_score_tau: float = 1.0,
) -> dict[str, Any]:
    """Compare persistence signatures with standard diagram-style distances."""

    if persistence_score_tau <= 0.0:
        raise ValueError("persistence_score_tau must be positive.")

    truth_h0 = [float(pair[1]) for pair in truth_diagrams.get("h0_finite", [])]
    pred_h0 = [float(pair[1]) for pair in pred_diagrams.get("h0_finite", [])]
    truth_h1 = [float(pair[0]) for pair in truth_diagrams.get("h1_essential", [])]
    pred_h1 = [float(pair[0]) for pair in pred_diagrams.get("h1_essential", [])]

    h0_wasserstein = _diagram_wasserstein_1d(
        truth_h0,
        pred_h0,
        order=2,
        diagonal_cost_scale=0.5,
    )
    h0_bottleneck = _diagram_bottleneck_1d(
        truth_h0,
        pred_h0,
        diagonal_cost_scale=0.5,
    )
    # H1 classes are essential in this filtration. We therefore compare their
    # birth coordinates with Wasserstein and bottleneck distances on the line,
    # and report a separate essential-class count error.
    h1_birth_wasserstein = _diagram_wasserstein_1d(
        truth_h1,
        pred_h1,
        order=2,
        diagonal_cost_scale=1.0,
    )
    h1_birth_bottleneck = _diagram_bottleneck_1d(
        truth_h1,
        pred_h1,
        diagonal_cost_scale=1.0,
    )
    h1_count_error = abs(len(pred_h1) - len(truth_h1))
    weighted_distance = (
        0.5 * h0_wasserstein
        + 0.3 * h1_birth_wasserstein
        + 0.2 * float(h1_count_error)
    )
    persistence_score = math.exp(-weighted_distance / float(persistence_score_tau))

    return {
        "h0_death_count_error": len(pred_h0) - len(truth_h0),
        "h0_2_wasserstein": h0_wasserstein,
        "h0_bottleneck": h0_bottleneck,
        "h1_essential_count_error": len(pred_h1) - len(truth_h1),
        "h1_essential_count_absolute_error": h1_count_error,
        "h1_birth_coordinate_2_wasserstein": h1_birth_wasserstein,
        "h1_birth_coordinate_bottleneck": h1_birth_bottleneck,
        "persistence_weighted_distance": weighted_distance,
        "persistence_score_tau": float(persistence_score_tau),
        "persistence_score": persistence_score,
    }


def evaluate_full_graph_recovery(
    truth_graph: nx.Graph | nx.MultiGraph,
    pred_graph: nx.Graph | nx.MultiGraph,
    *,
    truth_positions: dict[int, np.ndarray] | None = None,
    pred_positions: dict[int, np.ndarray] | None = None,
    observed_points: np.ndarray | None = None,
    truth_already_reduced_multigraph: bool = False,
    pred_already_reduced_multigraph: bool = False,
    normalize_persistence_edge_lengths: bool = True,
    persistence_score_tau: float = 1.0,
    representation_radius_scale: float = 0.5,
) -> dict[str, Any]:
    """Run the current full-graph-recovery evaluator."""

    truth_reduced = normalize_graph_for_evaluation(
        truth_graph,
        positions=truth_positions,
        already_reduced_multigraph=truth_already_reduced_multigraph,
    )
    pred_reduced = normalize_graph_for_evaluation(
        pred_graph,
        positions=pred_positions,
        already_reduced_multigraph=pred_already_reduced_multigraph,
    )
    topology = _compare_graph_topology_on_reduced(
        truth_graph=truth_graph,
        pred_graph=pred_graph,
        truth_reduced=truth_reduced,
        pred_reduced=pred_reduced,
        truth_already_reduced_multigraph=truth_already_reduced_multigraph,
        pred_already_reduced_multigraph=pred_already_reduced_multigraph,
    )
    relaxed_topology = _compare_relaxed_topology_on_reduced(truth_reduced, pred_reduced)
    cycle_recovery = _compare_cycle_core_on_reduced(truth_reduced, pred_reduced)
    branch_recovery = _compare_branch_forest_on_reduced(truth_reduced, pred_reduced)
    count_recovery = _compute_count_recovery_on_reduced(
        truth_reduced=truth_reduced,
        pred_reduced=pred_reduced,
    )
    truth_diagrams = compute_reduced_multigraph_persistence_diagrams(
        truth_reduced,
        normalize_edge_lengths=normalize_persistence_edge_lengths,
    )
    pred_diagrams = compute_reduced_multigraph_persistence_diagrams(
        pred_reduced,
        normalize_edge_lengths=normalize_persistence_edge_lengths,
    )
    persistence = compare_persistence_diagrams(
        truth_diagrams,
        pred_diagrams,
        persistence_score_tau=persistence_score_tau,
    )
    representation_quality = _compute_representation_quality(
        pred_graph=pred_graph,
        pred_positions=pred_positions,
        observed_points=observed_points,
        truth_graph=truth_graph,
        truth_positions=truth_positions,
        radius_scale=representation_radius_scale,
    )

    return {
        "task": "full_graph_recovery",
        "primary_metric": "relaxed_topology_score",
        "headline_scores": {
            "topology_score": float(relaxed_topology["relaxed_topology_score"]),
            "persistence_score": float(persistence["persistence_score"]),
            "cycle_recovery_score": float(cycle_recovery["cycle_recovery_score"]),
            "branch_recovery_score": float(branch_recovery["branch_recovery_score"]),
        },
        "graph_edit_distance": float(topology["graph_edit_distance"]),
        "graph_edit_distance_exact": bool(topology["graph_edit_distance_exact"]),
        "topology": topology,
        "relaxed_topology": relaxed_topology,
        "cycle_recovery": cycle_recovery,
        "branch_recovery": branch_recovery,
        "count_recovery": count_recovery,
        "persistence": persistence,
        "representation_quality": representation_quality,
        "truth_persistence_diagrams": truth_diagrams,
        "pred_persistence_diagrams": pred_diagrams,
    }


def evaluate_dataset_predictions(
    dataset_dir: str | Path,
    predictions_dir: str | Path,
    *,
    num_bins: int = 8,
    normalize_persistence_edge_lengths: bool = True,
    persistence_score_tau: float = 1.0,
    representation_radius_scale: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a directory of predictions against a benchmark dataset manifest."""

    if num_bins < 1:
        raise ValueError("num_bins must be positive.")

    dataset_path = Path(dataset_dir)
    predictions_path = Path(predictions_dir)
    manifest_path = dataset_path / "manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"Missing dataset manifest: {manifest_path}")
    if not predictions_path.is_dir():
        raise ValueError(f"Predictions directory does not exist: {predictions_path}")

    manifest_entries = _load_manifest(manifest_path)
    per_sample_results: list[dict[str, Any]] = []

    for entry in manifest_entries:
        truth_payload = _load_truth_payload_from_npz(dataset_path / entry["npz_path"])
        with np.load(dataset_path / entry["npz_path"], allow_pickle=False) as data:
            observed_points = np.asarray(data["noisy_points"], dtype=float)
        prediction_path = _resolve_prediction_path(
            predictions_dir=predictions_path,
            manifest_entry=entry,
        )
        prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))

        truth_graph, truth_positions, truth_already_reduced_multigraph = build_graph_from_payload(
            truth_payload,
            positions_required_for_persistence=True,
        )
        pred_graph, pred_positions, pred_already_reduced_multigraph = build_graph_from_payload(
            prediction_payload,
            positions_required_for_persistence=True,
        )
        sample_eval = evaluate_full_graph_recovery(
            truth_graph=truth_graph,
            pred_graph=pred_graph,
            truth_positions=truth_positions,
            pred_positions=pred_positions,
            observed_points=observed_points,
            truth_already_reduced_multigraph=truth_already_reduced_multigraph,
            pred_already_reduced_multigraph=pred_already_reduced_multigraph,
            normalize_persistence_edge_lengths=normalize_persistence_edge_lengths,
            persistence_score_tau=persistence_score_tau,
            representation_radius_scale=representation_radius_scale,
        )
        per_sample_results.append(
            {
                "sample_id": entry["sample_id"],
                "difficulty": entry.get("difficulty"),
                "difficulty_coordinate": entry["difficulty_coordinate"],
                "prediction_path": str(prediction_path),
                "headline_scores": sample_eval["headline_scores"],
                "topology": sample_eval["topology"],
                "cycle_recovery": sample_eval["cycle_recovery"],
                "branch_recovery": sample_eval["branch_recovery"],
                "count_recovery": sample_eval["count_recovery"],
                "persistence": sample_eval["persistence"],
                "representation_quality": sample_eval["representation_quality"],
                "graph_edit_distance": sample_eval["topology"]["graph_edit_distance"],
                "graph_edit_distance_exact": sample_eval["topology"]["graph_edit_distance_exact"],
            }
        )

    return {
        "task": "full_graph_recovery",
        "num_samples": len(per_sample_results),
        "primary_metric": "topology_score",
        "summary": _aggregate_dataset_results(per_sample_results),
        "axis_reports": _build_axis_reports(per_sample_results, num_bins=num_bins),
        "per_sample": per_sample_results,
    }


def _compare_graph_topology_on_reduced(
    *,
    truth_graph: nx.Graph | nx.MultiGraph,
    pred_graph: nx.Graph | nx.MultiGraph,
    truth_reduced: nx.MultiGraph,
    pred_reduced: nx.MultiGraph,
    truth_already_reduced_multigraph: bool = False,
    pred_already_reduced_multigraph: bool = False,
) -> dict[str, Any]:
    truth_summary = _summarize_graph(
        truth_graph,
        truth_reduced,
        already_reduced_multigraph=truth_already_reduced_multigraph,
    )
    pred_summary = _summarize_graph(
        pred_graph,
        pred_reduced,
        already_reduced_multigraph=pred_already_reduced_multigraph,
    )
    count_errors = {
        key: int(pred_summary[key] - truth_summary[key])
        for key in truth_summary
        if key in pred_summary
    }
    return {
        "exact_match": bool(
            nx.is_isomorphic(
                _project_to_simple_graph(truth_graph),
                _project_to_simple_graph(pred_graph),
            )
        ),
        "exact_match_up_to_degree2_suppression": _multigraph_isomorphic(
            truth_reduced,
            pred_reduced,
        ),
        **_graph_edit_distance_on_reduced(truth_reduced, pred_reduced),
        "truth_summary": truth_summary,
        "pred_summary": pred_summary,
        "count_errors": count_errors,
    }


def _compare_relaxed_topology_on_reduced(
    truth_reduced: nx.MultiGraph,
    pred_reduced: nx.MultiGraph,
) -> dict[str, Any]:
    """Compare reduced graphs with cycle-encoding-tolerant topology counts."""

    truth_summary = _summarize_multigraph_for_relaxed_topology(truth_reduced)
    pred_summary = _summarize_multigraph_for_relaxed_topology(pred_reduced)
    count_errors = {
        key: int(pred_summary[key] - truth_summary[key])
        for key in truth_summary
        if key in pred_summary
    }
    component_count_match = truth_summary["num_connected_components"] == pred_summary["num_connected_components"]
    cycle_rank_match = truth_summary["betti_1"] == pred_summary["betti_1"]
    leaf_count_match = truth_summary["leaf_count"] == pred_summary["leaf_count"]
    branch_node_count_match = truth_summary["branch_node_count"] == pred_summary["branch_node_count"]
    branch_component_count_match = truth_summary["branch_component_count"] == pred_summary["branch_component_count"]
    cycle_core_component_count_match = (
        truth_summary["cycle_core_component_count"] == pred_summary["cycle_core_component_count"]
    )

    checks = [
        component_count_match,
        cycle_rank_match,
        leaf_count_match,
        branch_node_count_match,
        branch_component_count_match,
        cycle_core_component_count_match,
    ]
    return {
        "relaxed_topology_score": _boolean_mean(checks),
        "relaxed_topology_match": all(checks),
        "component_count_match": component_count_match,
        "cycle_rank_match": cycle_rank_match,
        "leaf_count_match": leaf_count_match,
        "branch_node_count_match": branch_node_count_match,
        "branch_component_count_match": branch_component_count_match,
        "cycle_core_component_count_match": cycle_core_component_count_match,
        "truth_summary": truth_summary,
        "pred_summary": pred_summary,
        "count_errors": count_errors,
    }


def _compare_cycle_core_on_reduced(
    truth_reduced: nx.MultiGraph,
    pred_reduced: nx.MultiGraph,
) -> dict[str, Any]:
    truth_core = _extract_cycle_core(truth_reduced)
    pred_core = _extract_cycle_core(pred_reduced)
    truth_summary = _summarize_multigraph(truth_core)
    pred_summary = _summarize_multigraph(pred_core)
    cycle_rank_match = truth_summary["betti_1"] == pred_summary["betti_1"]
    core_component_count_match = (
        truth_summary["num_connected_components"] == pred_summary["num_connected_components"]
    )
    core_presence_match = (truth_summary["num_edges"] > 0) == (pred_summary["num_edges"] > 0)
    strict_match = _multigraph_isomorphic(truth_core, pred_core)
    return {
        "cycle_core_match": cycle_rank_match and core_component_count_match and core_presence_match,
        "strict_cycle_core_match": strict_match,
        "cycle_recovery_score": _boolean_mean(
            [cycle_rank_match, core_component_count_match, core_presence_match]
        ),
        "cycle_rank_match": cycle_rank_match,
        "core_component_count_match": core_component_count_match,
        "core_presence_match": core_presence_match,
        "truth_cycle_core_summary": truth_summary,
        "pred_cycle_core_summary": pred_summary,
    }


def _compare_branch_forest_on_reduced(
    truth_reduced: nx.MultiGraph,
    pred_reduced: nx.MultiGraph,
) -> dict[str, Any]:
    truth_signature = _branch_forest_signature(truth_reduced)
    pred_signature = _branch_forest_signature(pred_reduced)
    truth_sorted = sorted(truth_signature.elements())
    pred_sorted = sorted(pred_signature.elements())
    branch_forest_match = truth_signature == pred_signature
    branch_count_match = sum(truth_signature.values()) == sum(pred_signature.values())
    return {
        "branch_forest_match": branch_forest_match,
        "branch_recovery_score": _boolean_mean([branch_count_match, branch_forest_match]),
        "branch_count_match": branch_count_match,
        "truth_branch_count": sum(truth_signature.values()),
        "pred_branch_count": sum(pred_signature.values()),
        "truth_branch_forest_signature": truth_sorted,
        "pred_branch_forest_signature": pred_sorted,
    }


def _compute_count_recovery_on_reduced(
    *,
    truth_reduced: nx.MultiGraph,
    pred_reduced: nx.MultiGraph,
) -> dict[str, Any]:
    truth_cycle_count = _graph_betti_1(truth_reduced)
    pred_cycle_count = _graph_betti_1(pred_reduced)
    truth_branch_count = _branch_component_count(truth_reduced)
    pred_branch_count = _branch_component_count(pred_reduced)
    truth_component_count = (
        nx.number_connected_components(truth_reduced) if truth_reduced.number_of_nodes() else 0
    )
    pred_component_count = (
        nx.number_connected_components(pred_reduced) if pred_reduced.number_of_nodes() else 0
    )
    component_count_error = int(pred_component_count - truth_component_count)
    return {
        "truth_num_components": int(truth_component_count),
        "pred_num_components": int(pred_component_count),
        "component_count_correct": truth_component_count == pred_component_count,
        "component_count_error": component_count_error,
        "absolute_component_count_error": abs(component_count_error),
        "component_preservation_score": math.exp(-abs(component_count_error)),
        "truth_cycle_count": truth_cycle_count,
        "pred_cycle_count": pred_cycle_count,
        "cycle_count_correct": truth_cycle_count == pred_cycle_count,
        "absolute_cycle_count_error": abs(pred_cycle_count - truth_cycle_count),
        "truth_branch_count": truth_branch_count,
        "pred_branch_count": pred_branch_count,
        "branch_count_correct": truth_branch_count == pred_branch_count,
        "absolute_branch_count_error": abs(pred_branch_count - truth_branch_count),
    }


def _extract_cycle_core(G: nx.MultiGraph) -> nx.MultiGraph:
    core = G.copy()
    changed = True
    while changed:
        changed = False
        removable = [node for node in core.nodes() if core.degree(node) < 2]
        if removable:
            core.remove_nodes_from(removable)
            changed = True
    return core


def _branch_forest_signature(G: nx.MultiGraph) -> Counter[str]:
    cycle_core = _extract_cycle_core(G)
    cycle_nodes = set(cycle_core.nodes())
    if not cycle_nodes:
        signature = Counter()
        for component_nodes in nx.connected_components(G):
            component = G.subgraph(component_nodes).copy()
            signature[_encode_unrooted_component(component)] += 1
        return signature

    branch_nodes = [node for node in G.nodes() if node not in cycle_nodes]
    if not branch_nodes:
        return Counter()

    remaining = G.subgraph(branch_nodes).copy()
    signature = Counter()
    for component_nodes in nx.connected_components(remaining):
        component = remaining.subgraph(component_nodes).copy()
        root_candidates = sorted(
            node
            for node in component.nodes()
            if any(neighbor in cycle_nodes for neighbor in G.neighbors(node))
        )
        anchor_count = len(
            {
                neighbor
                for node in component.nodes()
                for neighbor in G.neighbors(node)
                if neighbor in cycle_nodes
            }
        )
        if _graph_betti_1(component) == 0:
            root_encoding = ",".join(
                _encode_rooted_tree(component, root, parent=None)
                for root in root_candidates
            )
            component_signature = (
                f"anchors={anchor_count}|roots={len(root_candidates)}|"
                f"nodes={component.number_of_nodes()}|edges={component.number_of_edges()}|"
                f"shape={root_encoding}"
            )
        else:
            degree_sequence = ",".join(
                str(degree) for _, degree in sorted(component.degree(), key=lambda item: item[0])
            )
            component_signature = (
                f"non_tree|anchors={anchor_count}|roots={len(root_candidates)}|"
                f"nodes={component.number_of_nodes()}|edges={component.number_of_edges()}|"
                f"betti={_graph_betti_1(component)}|degrees={degree_sequence}"
            )
        signature[component_signature] += 1
    return signature


def _encode_unrooted_component(component: nx.MultiGraph) -> str:
    if component.number_of_nodes() == 0:
        return "empty"

    betti = _graph_betti_1(component)
    simple = _project_to_simple_graph(component)
    if betti == 0 and nx.is_tree(simple):
        centers = sorted(nx.center(simple))
        encodings = [
            _encode_rooted_tree(simple, center, parent=None)
            for center in centers
        ]
        return "tree|" + min(encodings)

    degree_sequence = ",".join(
        str(degree) for _, degree in sorted(component.degree(), key=lambda item: item[0])
    )
    return (
        f"component|nodes={component.number_of_nodes()}|"
        f"edges={component.number_of_edges()}|betti={betti}|degrees={degree_sequence}"
    )


def _encode_rooted_tree(
    tree: nx.MultiGraph,
    node: Any,
    parent: Any | None,
) -> str:
    children_encodings = []
    for neighbor in sorted(tree.neighbors(node)):
        if neighbor == parent:
            continue
        children_encodings.append(_encode_rooted_tree(tree, neighbor, node))
    children_encodings.sort()
    return "(" + "".join(children_encodings) + ")"


def _graph_betti_1(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    return int(G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G))


def _branch_component_count(G: nx.MultiGraph) -> int:
    return sum(_branch_forest_signature(G).values())


def _multigraph_isomorphic(left: nx.MultiGraph, right: nx.MultiGraph) -> bool:
    matcher = nx.algorithms.isomorphism.MultiGraphMatcher(left, right)
    return bool(matcher.is_isomorphic())


def _project_multigraph_for_ged(G: nx.MultiGraph) -> nx.Graph:
    projected = nx.Graph()
    projected.add_nodes_from(G.nodes())
    multiplicities: dict[tuple[Any, Any], int] = {}
    for u, v in G.edges():
        key = (u, v) if str(u) <= str(v) else (v, u)
        multiplicities[key] = multiplicities.get(key, 0) + 1
    for (u, v), multiplicity in multiplicities.items():
        projected.add_edge(u, v, multiplicity=int(multiplicity))
    return projected


def _graph_edit_distance_on_reduced(
    truth_reduced: nx.MultiGraph,
    pred_reduced: nx.MultiGraph,
    *,
    timeout: float = 0.5,
) -> dict[str, Any]:
    truth_projected = _project_multigraph_for_ged(truth_reduced)
    pred_projected = _project_multigraph_for_ged(pred_reduced)

    if truth_projected.number_of_nodes() == 0 and pred_projected.number_of_nodes() == 0:
        return {
            "graph_edit_distance": 0.0,
            "graph_edit_distance_exact": True,
        }

    def node_del_cost(_: Any) -> float:
        return 1.0

    def node_ins_cost(_: Any) -> float:
        return 1.0

    def node_subst_cost(_: Any, __: Any) -> float:
        return 0.0

    def edge_del_cost(edge_data: dict[str, Any]) -> float:
        return float(edge_data.get("multiplicity", 1))

    def edge_ins_cost(edge_data: dict[str, Any]) -> float:
        return float(edge_data.get("multiplicity", 1))

    def edge_subst_cost(left: dict[str, Any], right: dict[str, Any]) -> float:
        return float(abs(left.get("multiplicity", 1) - right.get("multiplicity", 1)))

    ged = nx.graph_edit_distance(
        truth_projected,
        pred_projected,
        node_subst_cost=node_subst_cost,
        node_del_cost=node_del_cost,
        node_ins_cost=node_ins_cost,
        edge_subst_cost=edge_subst_cost,
        edge_del_cost=edge_del_cost,
        edge_ins_cost=edge_ins_cost,
        timeout=timeout,
    )

    if ged is None:
        truth_summary = _summarize_multigraph(truth_reduced)
        pred_summary = _summarize_multigraph(pred_reduced)
        ged = float(
            abs(truth_summary["num_nodes"] - pred_summary["num_nodes"])
            + abs(truth_summary["num_edges"] - pred_summary["num_edges"])
            + abs(truth_summary["self_loop_count"] - pred_summary["self_loop_count"])
            + abs(truth_summary["parallel_edge_surplus"] - pred_summary["parallel_edge_surplus"])
        )
        return {
            "graph_edit_distance": ged,
            "graph_edit_distance_exact": False,
        }

    return {
        "graph_edit_distance": float(ged),
        "graph_edit_distance_exact": True,
    }


def _summarize_multigraph(G: nx.MultiGraph) -> dict[str, Any]:
    return {
        "num_nodes": int(G.number_of_nodes()),
        "num_edges": int(G.number_of_edges()),
        "num_connected_components": int(nx.number_connected_components(G)) if G.number_of_nodes() else 0,
        "betti_1": _graph_betti_1(G),
        "self_loop_count": int(nx.number_of_selfloops(G)),
        "parallel_edge_surplus": int(_parallel_edge_surplus(G)),
    }


def _summarize_multigraph_for_relaxed_topology(G: nx.MultiGraph) -> dict[str, Any]:
    cycle_core = _extract_cycle_core(G)
    return {
        **_summarize_multigraph(G),
        "leaf_count": int(leaf_count(G)),
        "branch_node_count": int(len(branch_nodes(G))),
        "branch_component_count": int(_branch_component_count(G)),
        "cycle_core_component_count": (
            int(nx.number_connected_components(cycle_core))
            if cycle_core.number_of_nodes()
            else 0
        ),
        "cycle_core_betti_1": int(_graph_betti_1(cycle_core)),
    }


def _aggregate_dataset_results(
    per_sample_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "headline_scores": {
            "topology_score": _safe_mean(
                sample["headline_scores"]["topology_score"] for sample in per_sample_results
            ),
            "persistence_score": _safe_mean(
                sample["headline_scores"]["persistence_score"] for sample in per_sample_results
            ),
            "cycle_recovery_score": _safe_mean(
                sample["headline_scores"]["cycle_recovery_score"] for sample in per_sample_results
            ),
            "branch_recovery_score": _safe_mean(
                sample["headline_scores"]["branch_recovery_score"] for sample in per_sample_results
            ),
        },
        "diagnostics": {
            "mean_graph_edit_distance": _safe_mean(
                sample["topology"]["graph_edit_distance"] for sample in per_sample_results
            ),
            "graph_edit_distance_exact_rate": _safe_mean(
                float(sample["topology"]["graph_edit_distance_exact"]) for sample in per_sample_results
            ),
        },
        "representation_scores": {
            "mean_point_to_graph_score": _safe_mean(
                sample["representation_quality"]["point_to_graph_score"]
                for sample in per_sample_results
            ),
            "mean_coverage_score": _safe_mean(
                sample["representation_quality"]["coverage_score"]
                for sample in per_sample_results
            ),
            "mean_support_score": _safe_mean(
                sample["representation_quality"]["support_score"]
                for sample in per_sample_results
            ),
            "mean_unsupported_graph_mass_ratio": _safe_mean(
                sample["representation_quality"]["unsupported_graph_mass_ratio"]
                for sample in per_sample_results
            ),
            "mean_normalized_point_to_graph_distance": _safe_mean(
                sample["representation_quality"]["normalized_mean_point_to_graph_distance"]
                for sample in per_sample_results
            ),
        },
        "diagnostics": {
            "raw_exact_match_rate": _safe_mean(
                float(sample["topology"]["exact_match"]) for sample in per_sample_results
            ),
            "cycle_count_recovery": _safe_mean(
                float(sample["count_recovery"]["cycle_count_correct"]) for sample in per_sample_results
            ),
            "branch_count_recovery": _safe_mean(
                float(sample["count_recovery"]["branch_count_correct"]) for sample in per_sample_results
            ),
            "component_count_recovery": _safe_mean(
                float(sample["count_recovery"]["component_count_correct"]) for sample in per_sample_results
            ),
            "mean_component_preservation_score": _safe_mean(
                sample["count_recovery"]["component_preservation_score"] for sample in per_sample_results
            ),
            "mean_absolute_cycle_count_error": _safe_mean(
                sample["count_recovery"]["absolute_cycle_count_error"] for sample in per_sample_results
            ),
            "mean_absolute_branch_count_error": _safe_mean(
                sample["count_recovery"]["absolute_branch_count_error"] for sample in per_sample_results
            ),
            "mean_absolute_component_count_error": _safe_mean(
                sample["count_recovery"]["absolute_component_count_error"] for sample in per_sample_results
            ),
            "mean_h0_2_wasserstein": _safe_mean(
                sample["persistence"]["h0_2_wasserstein"] for sample in per_sample_results
            ),
            "mean_h0_bottleneck": _safe_mean(
                sample["persistence"]["h0_bottleneck"] for sample in per_sample_results
            ),
            "mean_h1_birth_coordinate_2_wasserstein": _safe_mean(
                sample["persistence"]["h1_birth_coordinate_2_wasserstein"]
                for sample in per_sample_results
            ),
            "mean_h1_birth_coordinate_bottleneck": _safe_mean(
                sample["persistence"]["h1_birth_coordinate_bottleneck"]
                for sample in per_sample_results
            ),
            "mean_h1_essential_count_absolute_error": _safe_mean(
                sample["persistence"]["h1_essential_count_absolute_error"]
                for sample in per_sample_results
            ),
            "mean_persistence_weighted_distance": _safe_mean(
                sample["persistence"]["persistence_weighted_distance"]
                for sample in per_sample_results
            ),
        },
        "representation_diagnostics": {
            "mean_graph_total_length_over_feature_scale": _safe_mean(
                sample["representation_quality"]["graph_total_length_over_feature_scale"]
                for sample in per_sample_results
            ),
            "mean_pred_num_nodes_with_geometry": _safe_mean(
                sample["representation_quality"]["pred_num_nodes"]
                for sample in per_sample_results
            ),
            "mean_pred_num_edges_with_geometry": _safe_mean(
                sample["representation_quality"]["pred_num_edges"]
                for sample in per_sample_results
            ),
        },
    }


def _build_axis_reports(
    per_sample_results: list[dict[str, Any]],
    *,
    num_bins: int,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for axis in (
        "noise_ratio",
        "crowding_r",
        "sampling_density_per_feature",
        "feature_thickness_ratio",
        "num_components",
        "requested_embedding_dimension",
    ):
        bins: list[list[dict[str, Any]]] = [[] for _ in range(num_bins)]
        for sample in per_sample_results:
            if axis not in sample["difficulty_coordinate"]:
                continue
            value = float(sample["difficulty_coordinate"][axis])
            hardness = _normalize_difficulty_coordinate(axis, value)
            bin_index = min(int(hardness * num_bins), num_bins - 1)
            bins[bin_index].append(sample)

        bin_reports = []
        for idx, bucket in enumerate(bins):
            bin_reports.append(
                {
                    "bin_index": idx,
                    "normalized_hardness_range": [idx / num_bins, (idx + 1) / num_bins],
                    "sample_count": len(bucket),
                    "headline_scores": {
                        "topology_score": _safe_mean(
                            sample["headline_scores"]["topology_score"] for sample in bucket
                        ),
                        "persistence_score": _safe_mean(
                            sample["headline_scores"]["persistence_score"] for sample in bucket
                        ),
                        "cycle_recovery_score": _safe_mean(
                            sample["headline_scores"]["cycle_recovery_score"] for sample in bucket
                        ),
                        "branch_recovery_score": _safe_mean(
                            sample["headline_scores"]["branch_recovery_score"] for sample in bucket
                        ),
                    },
                    "mean_graph_edit_distance": _safe_mean(
                        sample["topology"]["graph_edit_distance"] for sample in bucket
                    ),
                    "count_recovery": {
                        "cycle_count_recovery": _safe_mean(
                            float(sample["count_recovery"]["cycle_count_correct"])
                            for sample in bucket
                        ),
                        "branch_count_recovery": _safe_mean(
                            float(sample["count_recovery"]["branch_count_correct"])
                            for sample in bucket
                        ),
                        "component_count_recovery": _safe_mean(
                            float(sample["count_recovery"]["component_count_correct"])
                            for sample in bucket
                        ),
                        "component_preservation_score": _safe_mean(
                            sample["count_recovery"]["component_preservation_score"]
                            for sample in bucket
                        ),
                    },
                    "mean_persistence_weighted_distance": _safe_mean(
                        sample["persistence"]["persistence_weighted_distance"]
                        for sample in bucket
                    ),
                }
            )
        reports[axis] = {
            "bins": bin_reports,
            "axis_sensitivity": _axis_sensitivity(bin_reports),
        }
    return reports


def _axis_sensitivity(bin_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not bin_reports:
        return {}
    easy = next((item for item in bin_reports if item["sample_count"] > 0), None)
    hard = next((item for item in reversed(bin_reports) if item["sample_count"] > 0), None)
    if easy is None or hard is None:
        return {}
    return {
        "topology_drop_easy_to_hard": _difference(
            easy["headline_scores"]["topology_score"],
            hard["headline_scores"]["topology_score"],
        ),
        "persistence_drop_easy_to_hard": _difference(
            easy["headline_scores"]["persistence_score"],
            hard["headline_scores"]["persistence_score"],
        ),
        "cycle_recovery_drop_easy_to_hard": _difference(
            easy["headline_scores"]["cycle_recovery_score"],
            hard["headline_scores"]["cycle_recovery_score"],
        ),
        "branch_recovery_drop_easy_to_hard": _difference(
            easy["headline_scores"]["branch_recovery_score"],
            hard["headline_scores"]["branch_recovery_score"],
        ),
        "cycle_count_recovery_drop_easy_to_hard": _difference(
            easy["count_recovery"]["cycle_count_recovery"],
            hard["count_recovery"]["cycle_count_recovery"],
        ),
        "branch_count_recovery_drop_easy_to_hard": _difference(
            easy["count_recovery"]["branch_count_recovery"],
            hard["count_recovery"]["branch_count_recovery"],
        ),
    }


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _safe_mean(values: Any) -> float | None:
    values_list = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    if not values_list:
        return None
    return float(sum(values_list) / len(values_list))


def _boolean_mean(values: list[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(1.0 for value in values if value) / len(values))


def _compute_representation_quality(
    *,
    pred_graph: nx.Graph | nx.MultiGraph,
    pred_positions: dict[int, np.ndarray] | None,
    observed_points: np.ndarray | None,
    truth_graph: nx.Graph | nx.MultiGraph,
    truth_positions: dict[int, np.ndarray] | None,
    radius_scale: float,
) -> dict[str, Any]:
    if observed_points is None or pred_positions is None:
        return {
            "available": False,
            "reason": "missing_observed_points_or_prediction_geometry",
            "point_to_graph_score": float("nan"),
            "coverage_score": float("nan"),
            "support_score": float("nan"),
            "unsupported_graph_mass_ratio": float("nan"),
            "normalized_mean_point_to_graph_distance": float("nan"),
            "graph_total_length_over_feature_scale": float("nan"),
            "pred_num_nodes": int(pred_graph.number_of_nodes()),
            "pred_num_edges": int(pred_graph.number_of_edges()),
        }

    observed_points = np.asarray(observed_points, dtype=float)
    pred_dims = {np.asarray(coords, dtype=float).shape[0] for coords in pred_positions.values()}
    if len(pred_dims) != 1 or observed_points.shape[1] != next(iter(pred_dims)):
        return {
            "available": False,
            "reason": "geometry_dimension_mismatch",
            "point_to_graph_score": float("nan"),
            "coverage_score": float("nan"),
            "support_score": float("nan"),
            "unsupported_graph_mass_ratio": float("nan"),
            "normalized_mean_point_to_graph_distance": float("nan"),
            "graph_total_length_over_feature_scale": float("nan"),
            "pred_num_nodes": int(pred_graph.number_of_nodes()),
            "pred_num_edges": int(pred_graph.number_of_edges()),
        }
    feature_scale = _feature_scale_from_truth_graph(truth_graph, truth_positions)
    if not np.isfinite(feature_scale) or feature_scale <= 0.0:
        feature_scale = 1.0
    radius = float(radius_scale) * float(feature_scale)

    point_distances = _point_to_graph_distances(observed_points, pred_graph, pred_positions)
    mean_distance = float(np.mean(point_distances)) if point_distances.size else 0.0
    normalized_mean_distance = float(mean_distance / feature_scale)
    coverage_score = float(np.mean(point_distances <= radius)) if point_distances.size else 0.0

    graph_sample_points, graph_total_length = _sample_graph_geometry(
        pred_graph,
        pred_positions,
        feature_scale=feature_scale,
        radius=radius,
    )
    if graph_sample_points.size == 0:
        unsupported_ratio = 1.0
        support_score = 0.0
    else:
        tree = cKDTree(observed_points)
        support_distances, _ = tree.query(graph_sample_points, k=1)
        unsupported_ratio = float(np.mean(np.asarray(support_distances, dtype=float) > radius))
        support_score = float(1.0 - unsupported_ratio)

    return {
        "available": True,
        "radius_scale": float(radius_scale),
        "support_radius": float(radius),
        "feature_scale": float(feature_scale),
        "mean_point_to_graph_distance": mean_distance,
        "normalized_mean_point_to_graph_distance": normalized_mean_distance,
        "point_to_graph_score": float(math.exp(-normalized_mean_distance)),
        "coverage_score": coverage_score,
        "support_score": support_score,
        "unsupported_graph_mass_ratio": unsupported_ratio,
        "graph_total_length": float(graph_total_length),
        "graph_total_length_over_feature_scale": float(graph_total_length / feature_scale),
        "graph_num_sample_points": int(graph_sample_points.shape[0]),
        "pred_num_nodes": int(pred_graph.number_of_nodes()),
        "pred_num_edges": int(pred_graph.number_of_edges()),
    }


def _feature_scale_from_truth_graph(
    truth_graph: nx.Graph | nx.MultiGraph,
    positions: dict[int, np.ndarray] | None,
) -> float:
    if not positions or truth_graph.number_of_edges() == 0:
        return 1.0
    lengths = []
    for u, v in truth_graph.edges():
        if u in positions and v in positions:
            lengths.append(float(np.linalg.norm(np.asarray(positions[u]) - np.asarray(positions[v]))))
    finite = [length for length in lengths if np.isfinite(length) and length > 0.0]
    if finite:
        return float(np.median(np.asarray(finite, dtype=float)))
    return 1.0


def _point_to_graph_distances(
    points: np.ndarray,
    G: nx.Graph | nx.MultiGraph,
    positions: dict[int, np.ndarray],
) -> np.ndarray:
    if points.size == 0:
        return np.zeros(0, dtype=float)

    node_positions = np.asarray([positions[node] for node in G.nodes() if node in positions], dtype=float)
    if node_positions.size == 0:
        return np.full(points.shape[0], np.inf, dtype=float)

    tree = cKDTree(node_positions)
    min_distances, _ = tree.query(points, k=1)
    min_distances = np.asarray(min_distances, dtype=float)

    edge_iter = G.edges(keys=True) if isinstance(G, nx.MultiGraph) else G.edges()
    for edge in edge_iter:
        u, v = edge[:2]
        if u not in positions or v not in positions:
            continue
        seg_distances = _point_to_segment_distances(
            points,
            np.asarray(positions[u], dtype=float),
            np.asarray(positions[v], dtype=float),
        )
        min_distances = np.minimum(min_distances, seg_distances)
    return min_distances


def _point_to_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = end - start
    denom = float(np.dot(direction, direction))
    if denom <= np.finfo(float).eps:
        return np.linalg.norm(points - start[None, :], axis=1)
    t = np.clip(((points - start[None, :]) @ direction) / denom, 0.0, 1.0)
    projections = start[None, :] + t[:, None] * direction[None, :]
    return np.linalg.norm(points - projections, axis=1)


def _sample_graph_geometry(
    G: nx.Graph | nx.MultiGraph,
    positions: dict[int, np.ndarray],
    *,
    feature_scale: float,
    radius: float,
) -> tuple[np.ndarray, float]:
    samples: list[np.ndarray] = []
    total_length = 0.0

    edge_iter = G.edges(keys=True) if isinstance(G, nx.MultiGraph) else G.edges()
    for edge in edge_iter:
        u, v = edge[:2]
        if u not in positions or v not in positions:
            continue
        start = np.asarray(positions[u], dtype=float)
        end = np.asarray(positions[v], dtype=float)
        length = float(np.linalg.norm(end - start))
        total_length += length
        if length <= np.finfo(float).eps:
            samples.append(start[None, :])
            continue
        step = max(radius * 0.5, feature_scale * 0.1, np.finfo(float).eps)
        num = max(2, int(math.ceil(length / step)) + 1)
        ts = np.linspace(0.0, 1.0, num=num, dtype=float)
        samples.append(start[None, :] + ts[:, None] * (end - start)[None, :])

    if samples:
        return np.vstack(samples), total_length

    node_samples = [np.asarray(positions[node], dtype=float) for node in G.nodes() if node in positions]
    if not node_samples:
        return np.empty((0, 0), dtype=float), 0.0
    return np.vstack(node_samples), 0.0


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                entries.append(json.loads(stripped))
    return entries


def _resolve_prediction_path(
    *,
    predictions_dir: Path,
    manifest_entry: dict[str, Any],
) -> Path:
    stem = Path(manifest_entry["npz_path"]).stem
    sample_id = int(manifest_entry["sample_id"])
    candidates = [
        predictions_dir / f"{stem}.json",
        predictions_dir / f"sample_{sample_id:05d}.json",
        predictions_dir / f"{sample_id}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"Could not find a prediction JSON for sample_id={sample_id} in {predictions_dir}."
    )


def _normalize_difficulty_coordinate(axis: str, value: float) -> float:
    if axis == "noise_ratio":
        return _normalize_interval(value, low=0.02, high=0.20, invert=False)
    if axis == "crowding_r":
        return _normalize_interval(value, low=0.00, high=0.60, invert=False)
    if axis == "sampling_density_per_feature":
        return _normalize_interval(value, low=3.5, high=25.0, invert=True)
    if axis == "feature_thickness_ratio":
        return _normalize_interval(value, low=0.45, high=1.00, invert=False)
    if axis == "num_components":
        return _normalize_interval(value, low=1.0, high=4.0, invert=False)
    if axis == "requested_embedding_dimension":
        return _normalize_interval(value, low=2.0, high=5.0, invert=False)
    raise ValueError(f"Unsupported difficulty axis: {axis}")


def _normalize_interval(
    value: float,
    *,
    low: float,
    high: float,
    invert: bool,
) -> float:
    if high <= low:
        raise ValueError("Expected high > low for interval normalization.")
    clipped = min(max(float(value), low), high)
    fraction = (clipped - low) / (high - low)
    if invert:
        fraction = 1.0 - fraction
    return float(min(max(fraction, 0.0), 1.0))


def _parse_positions(
    payload: dict[str, Any],
    nodes: list[Any],
) -> dict[int, np.ndarray] | None:
    raw_positions = payload.get("node_positions")
    if raw_positions is None:
        return None

    if isinstance(raw_positions, dict):
        parsed: dict[int, np.ndarray] = {}
        for node in nodes:
            if node in raw_positions:
                coords = raw_positions[node]
            elif str(node) in raw_positions:
                coords = raw_positions[str(node)]
            else:
                raise ValueError(f"node_positions is missing coordinates for node {node!r}.")
            parsed[node] = np.asarray(coords, dtype=float)
        return parsed

    if isinstance(raw_positions, list):
        if len(raw_positions) != len(nodes):
            raise ValueError("List-based node_positions must align one-to-one with nodes.")
        return {
            node: np.asarray(coords, dtype=float)
            for node, coords in zip(nodes, raw_positions)
        }

    raise ValueError("node_positions must be either a dict or a list.")


def _attach_edge_lengths(
    G: nx.Graph | nx.MultiGraph,
    edges: list[tuple[Any, Any, Any | None]],
    edge_lengths: Any,
) -> None:
    if isinstance(edge_lengths, list):
        if len(edge_lengths) != len(edges):
            raise ValueError("List-based edge_lengths must align one-to-one with edges.")
        for edge, length in zip(edges, edge_lengths):
            u, v, key = edge
            if isinstance(G, nx.MultiGraph):
                if key is None:
                    raise ValueError("Multigraph edge lengths require stable edge keys.")
                G[u][v][key]["length"] = float(length)
            else:
                G[u][v]["length"] = float(length)
        return

    if isinstance(edge_lengths, dict):
        for u, v, key in edges:
            lookup_candidates = []
            if key is not None:
                lookup_candidates.extend([str(key), key])
            lookup_candidates.extend(
                [
                    f"{u},{v}",
                    f"{v},{u}",
                    str([u, v]),
                    str([v, u]),
                ]
            )
            for candidate in lookup_candidates:
                if candidate in edge_lengths:
                    length = edge_lengths[candidate]
                    break
            else:
                raise ValueError(f"edge_lengths is missing an entry for edge {(u, v)!r}.")
            if isinstance(G, nx.MultiGraph):
                if key is None:
                    raise ValueError("Multigraph edge lengths require stable edge keys.")
                G[u][v][key]["length"] = float(length)
            else:
                G[u][v]["length"] = float(length)
        return

    raise ValueError("edge_lengths must be either a dict or a list.")


def _graph_has_edge_lengths(G: nx.Graph | nx.MultiGraph) -> bool:
    if isinstance(G, nx.MultiGraph):
        return all("length" in data for _, _, _, data in G.edges(keys=True, data=True))
    return all("length" in data for _, _, data in G.edges(data=True))


def _resolve_edge_lengths(
    G: nx.Graph,
    positions: dict[int, np.ndarray],
) -> dict[tuple[Any, Any], float]:
    edge_lengths: dict[tuple[Any, Any], float] = {}
    for u, v, data in G.edges(data=True):
        edge = _edge_key(u, v)
        if "length" in data:
            edge_lengths[edge] = float(data["length"])
            continue
        if u in positions and v in positions:
            edge_lengths[edge] = float(np.linalg.norm(positions[u] - positions[v]))
            continue
        edge_lengths[edge] = 1.0
    return edge_lengths


def _summarize_graph(
    G: nx.Graph | nx.MultiGraph,
    reduced: nx.MultiGraph,
    *,
    already_reduced_multigraph: bool = False,
) -> dict[str, Any]:
    graph_for_counts = _project_to_simple_graph(G) if isinstance(G, nx.MultiGraph) else G
    summary_num_nodes = int(G.number_of_nodes())
    summary_num_edges = int(G.number_of_edges())
    summary_num_components = (
        int(nx.number_connected_components(G)) if G.number_of_nodes() else 0
    )
    summary_betti_1 = int(_graph_betti_1(G))
    if already_reduced_multigraph:
        graph_for_counts = _project_to_simple_graph(reduced)
        summary_num_nodes = int(reduced.number_of_nodes())
        summary_num_edges = int(reduced.number_of_edges())
        summary_num_components = (
            int(nx.number_connected_components(reduced)) if reduced.number_of_nodes() else 0
        )
        summary_betti_1 = int(_graph_betti_1(reduced))
    return {
        "num_nodes": summary_num_nodes,
        "num_edges": summary_num_edges,
        "num_connected_components": summary_num_components,
        "betti_1": summary_betti_1,
        "cycle_count": int(_graph_betti_1(reduced)),
        "branch_count": int(_branch_component_count(reduced)),
        "leaf_count": int(leaf_count(graph_for_counts)),
        "branch_node_count": int(len(branch_nodes(graph_for_counts))),
        "reduced_num_nodes": int(reduced.number_of_nodes()),
        "reduced_num_edges": int(reduced.number_of_edges()),
        "reduced_self_loop_count": int(nx.number_of_selfloops(reduced)),
        "reduced_parallel_edge_surplus": int(_parallel_edge_surplus(reduced)),
    }


def normalize_graph_for_evaluation(
    G: nx.Graph | nx.MultiGraph,
    *,
    positions: dict[int, np.ndarray] | None = None,
    already_reduced_multigraph: bool = False,
) -> nx.MultiGraph:
    if already_reduced_multigraph:
        return canonicalize_reduced_multigraph(G, positions=positions)
    return reduce_graph_topology(G, positions=positions)


def canonicalize_reduced_multigraph(
    G: nx.Graph | nx.MultiGraph,
    *,
    positions: dict[int, np.ndarray] | None = None,
) -> nx.MultiGraph:
    if G.is_directed():
        raise ValueError("Only undirected graphs are supported.")

    reduced = nx.MultiGraph()
    for node, data in G.nodes(data=True):
        reduced.add_node(node, **dict(data))

    if isinstance(G, nx.MultiGraph):
        for u, v, key, data in G.edges(keys=True, data=True):
            edge_data = dict(data)
            if "total_length" not in edge_data and "length" not in edge_data and positions is not None:
                if u in positions and v in positions:
                    edge_data["length"] = float(np.linalg.norm(positions[u] - positions[v]))
            reduced.add_edge(u, v, key=key, **edge_data)
        return reduced

    for edge_index, (u, v, data) in enumerate(G.edges(data=True)):
        edge_data = dict(data)
        if "total_length" not in edge_data and "length" not in edge_data and positions is not None:
            if u in positions and v in positions:
                edge_data["length"] = float(np.linalg.norm(positions[u] - positions[v]))
        reduced.add_edge(u, v, key=edge_index, **edge_data)
    return reduced


def _build_exact_graph_affinity_matrix(
    G: nx.Graph,
    *,
    positions: dict[int, np.ndarray] | None = None,
) -> tuple[sp.csr_matrix, list[Any]]:
    node_order = sorted(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}
    n = len(node_order)
    if n == 0:
        return sp.csr_matrix((0, 0), dtype=float), node_order

    edge_lengths = _resolve_edge_lengths(G, positions or {})
    local_scales = np.ones(n, dtype=float)
    incident_lengths: list[list[float]] = [[] for _ in range(n)]
    for u, v in G.edges():
        length = float(edge_lengths[_edge_key(u, v)])
        incident_lengths[node_to_idx[u]].append(length)
        incident_lengths[node_to_idx[v]].append(length)

    for idx, lengths in enumerate(incident_lengths):
        if lengths:
            local_scales[idx] = max(float(max(lengths)), np.finfo(float).eps)

    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for u, v in G.edges():
        i = node_to_idx[u]
        j = node_to_idx[v]
        length = float(edge_lengths[_edge_key(u, v)])
        denom = max(local_scales[i] * local_scales[j], np.finfo(float).eps)
        weight = float(math.exp(-(length ** 2) / denom))
        rows.extend([i, j])
        cols.extend([j, i])
        weights.extend([weight, weight])

    W = sp.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=float)
    W = W.maximum(W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W, node_order


def _project_to_simple_graph(G: nx.Graph | nx.MultiGraph) -> nx.Graph:
    if not isinstance(G, nx.MultiGraph):
        return G
    simple = nx.Graph()
    simple.add_nodes_from(G.nodes(data=True))
    simple.add_edges_from((u, v) for u, v in G.edges())
    return simple


def _parallel_edge_surplus(G: nx.MultiGraph) -> int:
    surplus = 0
    seen_pairs: set[tuple[Any, Any]] = set()
    for u, v, _ in G.edges(keys=True):
        pair = _edge_key(u, v)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        edge_count = G.number_of_edges(*pair)
        surplus += max(edge_count - 1, 0)
    return surplus


def _normalize_lengths(
    lengths: list[float],
    *,
    normalize_edge_lengths: bool,
) -> tuple[list[float], float]:
    if not lengths:
        return [], 1.0
    scale = 1.0
    if normalize_edge_lengths:
        positive_lengths = [value for value in lengths if value > 0.0 and math.isfinite(value)]
        if positive_lengths:
            scale = float(np.median(np.asarray(positive_lengths, dtype=float)))
            if scale <= 0.0 or not math.isfinite(scale):
                scale = 1.0
    return [float(value / scale) for value in lengths], float(scale)


def _diagram_wasserstein_1d(
    truth_values: list[float],
    pred_values: list[float],
    *,
    order: int,
    diagonal_cost_scale: float,
) -> float:
    if order < 1:
        raise ValueError("order must be positive.")
    truth_sorted = sorted(float(value) for value in truth_values)
    pred_sorted = sorted(float(value) for value in pred_values)
    rows = len(truth_sorted) + 1
    cols = len(pred_sorted) + 1
    dp = [[math.inf] * cols for _ in range(rows)]
    dp[0][0] = 0.0

    for i in range(rows):
        for j in range(cols):
            current = dp[i][j]
            if not math.isfinite(current):
                continue
            if i < len(truth_sorted):
                diag_cost = (diagonal_cost_scale * abs(truth_sorted[i])) ** order
                dp[i + 1][j] = min(dp[i + 1][j], current + diag_cost)
            if j < len(pred_sorted):
                diag_cost = (diagonal_cost_scale * abs(pred_sorted[j])) ** order
                dp[i][j + 1] = min(dp[i][j + 1], current + diag_cost)
            if i < len(truth_sorted) and j < len(pred_sorted):
                pair_cost = abs(truth_sorted[i] - pred_sorted[j]) ** order
                dp[i + 1][j + 1] = min(dp[i + 1][j + 1], current + pair_cost)
    return float(dp[-1][-1] ** (1.0 / order))


def _diagram_bottleneck_1d(
    truth_values: list[float],
    pred_values: list[float],
    *,
    diagonal_cost_scale: float,
) -> float:
    truth_sorted = sorted(float(value) for value in truth_values)
    pred_sorted = sorted(float(value) for value in pred_values)
    candidates = {0.0}
    for value in truth_sorted:
        candidates.add(diagonal_cost_scale * abs(value))
    for value in pred_sorted:
        candidates.add(diagonal_cost_scale * abs(value))
    for left in truth_sorted:
        for right in pred_sorted:
            candidates.add(abs(left - right))

    ordered = sorted(candidates)
    low = 0
    high = len(ordered) - 1
    best = ordered[-1]
    while low <= high:
        mid = (low + high) // 2
        threshold = ordered[mid]
        if _diagram_matching_feasible(truth_sorted, pred_sorted, threshold, diagonal_cost_scale):
            best = threshold
            high = mid - 1
        else:
            low = mid + 1
    return float(best)


def _diagram_matching_feasible(
    truth_sorted: list[float],
    pred_sorted: list[float],
    threshold: float,
    diagonal_cost_scale: float,
) -> bool:
    rows = len(truth_sorted) + 1
    cols = len(pred_sorted) + 1
    reachable = [[False] * cols for _ in range(rows)]
    reachable[0][0] = True
    for i in range(rows):
        for j in range(cols):
            if not reachable[i][j]:
                continue
            if i < len(truth_sorted) and diagonal_cost_scale * abs(truth_sorted[i]) <= threshold:
                reachable[i + 1][j] = True
            if j < len(pred_sorted) and diagonal_cost_scale * abs(pred_sorted[j]) <= threshold:
                reachable[i][j + 1] = True
            if i < len(truth_sorted) and j < len(pred_sorted):
                if abs(truth_sorted[i] - pred_sorted[j]) <= threshold:
                    reachable[i + 1][j + 1] = True
    return bool(reachable[-1][-1])


def _edge_key(u: Any, v: Any) -> tuple[Any, Any]:
    return (u, v) if u <= v else (v, u)


def _load_truth_payload_from_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        nodes = data["nodes"].tolist()
        edges = data["edges"].tolist()
        positions_array = np.asarray(data["positions"], dtype=float)
    if len(nodes) != positions_array.shape[0]:
        raise ValueError("Truth .npz file has inconsistent nodes and positions arrays.")
    return {
        "nodes": nodes,
        "edges": edges,
        "node_positions": {
            str(node): positions_array[idx].tolist()
            for idx, node in enumerate(nodes)
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser(
        "sample",
        help="Evaluate one prediction against one ground-truth sample.",
    )
    sample_parser.add_argument(
        "--truth-npz",
        required=True,
        help="Ground-truth sample .npz file produced by batch_generate_benchmark.py.",
    )
    sample_parser.add_argument(
        "--prediction-json",
        required=True,
        help="Prediction JSON file containing recovered nodes, edges, and geometry.",
    )
    sample_parser.add_argument(
        "--no-normalize-persistence-edge-lengths",
        action="store_true",
        help="Disable per-graph edge-length normalization before persistence extraction.",
    )
    sample_parser.add_argument(
        "--persistence-score-tau",
        type=float,
        default=1.0,
        help="Positive scale parameter used when converting persistence distance into a score.",
    )

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Evaluate a prediction directory against a dataset manifest.",
    )
    dataset_parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing manifest.jsonl and the benchmark samples/ subdirectory.",
    )
    dataset_parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory containing one prediction JSON per sample.",
    )
    dataset_parser.add_argument(
        "--num-bins",
        type=int,
        default=8,
        help="Number of bins to use for each difficulty axis report.",
    )
    dataset_parser.add_argument(
        "--no-normalize-persistence-edge-lengths",
        action="store_true",
        help="Disable per-graph edge-length normalization before persistence extraction.",
    )
    dataset_parser.add_argument(
        "--persistence-score-tau",
        type=float,
        default=1.0,
        help="Positive scale parameter used when converting persistence distance into a score.",
    )
    dataset_parser.add_argument(
        "--output-path",
        default=None,
        help="Optional JSON path to write the full dataset evaluation report.",
    )
    dataset_parser.add_argument(
        "--manifold-output-path",
        default=None,
        help="Optional PNG path for the performance-colored difficulty manifold. If omitted and --output-path is set, a sibling PNG is written automatically.",
    )
    dataset_parser.add_argument(
        "--manifold-metadata-output-path",
        default=None,
        help="Optional JSON path for manifold matching metadata.",
    )
    dataset_parser.add_argument(
        "--skip-manifold-render",
        action="store_true",
        help="Disable automatic manifold rendering for dataset evaluation outputs.",
    )
    dataset_parser.add_argument(
        "--manifold-epsilon",
        type=float,
        default=None,
        help="Optional Gaussian affinity scale for the difficulty-manifold render.",
    )
    dataset_parser.add_argument(
        "--manifold-candidate-dims",
        type=int,
        default=12,
        help="How many nontrivial diffusion eigenvectors to consider when matching the manifold axes.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "sample":
        truth_payload = _load_truth_payload_from_npz(args.truth_npz)
        prediction_payload = json.loads(Path(args.prediction_json).read_text(encoding="utf-8"))
        truth_graph, truth_positions, truth_already_reduced_multigraph = build_graph_from_payload(
            truth_payload,
            positions_required_for_persistence=True,
        )
        pred_graph, pred_positions, pred_already_reduced_multigraph = build_graph_from_payload(
            prediction_payload,
            positions_required_for_persistence=True,
        )
        result = evaluate_full_graph_recovery(
            truth_graph=truth_graph,
            pred_graph=pred_graph,
            truth_positions=truth_positions,
            pred_positions=pred_positions,
            truth_already_reduced_multigraph=truth_already_reduced_multigraph,
            pred_already_reduced_multigraph=pred_already_reduced_multigraph,
            normalize_persistence_edge_lengths=not args.no_normalize_persistence_edge_lengths,
            persistence_score_tau=args.persistence_score_tau,
        )
    else:
        result = evaluate_dataset_predictions(
            dataset_dir=args.dataset_dir,
            predictions_dir=args.predictions_dir,
            num_bins=args.num_bins,
            normalize_persistence_edge_lengths=not args.no_normalize_persistence_edge_lengths,
            persistence_score_tau=args.persistence_score_tau,
        )
        if args.output_path is not None:
            _write_json(Path(args.output_path), result)
        if not args.skip_manifold_render:
            manifold_output_path = _resolve_manifold_output_path(
                report_output_path=args.output_path,
                manifold_output_path=args.manifold_output_path,
            )
            if manifold_output_path is not None:
                manifold_metadata_output_path = _resolve_manifold_metadata_output_path(
                    report_output_path=args.output_path,
                    manifold_output_path=manifold_output_path,
                    manifold_metadata_output_path=args.manifold_metadata_output_path,
                )
                _render_dataset_manifold(
                    report=result,
                    manifold_output_path=Path(manifold_output_path),
                    manifold_metadata_output_path=(
                        Path(manifold_metadata_output_path)
                        if manifold_metadata_output_path is not None
                        else None
                    ),
                    epsilon=args.manifold_epsilon,
                    candidate_dims=args.manifold_candidate_dims,
                )
    print(json.dumps(_to_jsonable(result), indent=2))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def _resolve_manifold_output_path(
    *,
    report_output_path: str | None,
    manifold_output_path: str | None,
) -> str | None:
    if manifold_output_path is not None:
        return manifold_output_path
    if report_output_path is None:
        return None
    report_path = Path(report_output_path)
    return str(report_path.with_name(f"{report_path.stem}_diffusion_manifold.png"))


def _resolve_manifold_metadata_output_path(
    *,
    report_output_path: str | None,
    manifold_output_path: str,
    manifold_metadata_output_path: str | None,
) -> str | None:
    if manifold_metadata_output_path is not None:
        return manifold_metadata_output_path
    if report_output_path is not None:
        report_path = Path(report_output_path)
        return str(report_path.with_name(f"{report_path.stem}_diffusion_manifold.json"))
    manifold_path = Path(manifold_output_path)
    return str(manifold_path.with_suffix(".json"))


def _render_dataset_manifold(
    *,
    report: dict[str, Any],
    manifold_output_path: Path,
    manifold_metadata_output_path: Path | None,
    epsilon: float | None,
    candidate_dims: int,
) -> None:
    from .render_diffusion_difficulty_manifold import render_diffusion_difficulty_manifold

    render_diffusion_difficulty_manifold(
        report_data=report,
        output_path=manifold_output_path,
        metadata_output_path=manifold_metadata_output_path,
        epsilon=epsilon,
        candidate_dims=candidate_dims,
        axis_only=False,
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
