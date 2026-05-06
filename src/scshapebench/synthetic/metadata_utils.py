"""Metadata helpers shared across topology, embedding, and sampling stages."""

from __future__ import annotations

from typing import Any

import networkx as nx

from .graph_primitives import _edge_key

__all__ = [
    "build_node_cycle_membership",
    "compute_backbone_nodes",
    "cycle_edge_lengths",
    "cycle_edge_membership",
    "extract_attachment_map_from_metadata",
    "extract_branch_children_from_metadata",
    "extract_cycle_ordering_from_metadata",
    "infer_edge_branch_depth",
    "infer_edge_branch_root",
    "infer_edge_cycle_id",
    "infer_node_cycle_id",
]


def cycle_edge_membership(
    cycle_nodes: dict[int, list[int]],
) -> dict[tuple[int, int], list[int]]:
    """Map each motif edge in the final graph to the cycle ids that contain it."""

    membership: dict[tuple[int, int], list[int]] = {}
    for cycle_id, nodes in cycle_nodes.items():
        for idx in range(len(nodes)):
            edge = _edge_key(nodes[idx], nodes[(idx + 1) % len(nodes)])
            membership.setdefault(edge, []).append(cycle_id)
    return membership


def cycle_edge_lengths(
    cycle_nodes: list[int],
    edge_lengths: dict[tuple[int, int], float],
) -> list[float]:
    """Return target edge lengths in cycle order."""

    return [
        edge_lengths[_edge_key(cycle_nodes[idx], cycle_nodes[(idx + 1) % len(cycle_nodes)])]
        for idx in range(len(cycle_nodes))
    ]


def build_node_cycle_membership(cycle_nodes: dict[int, list[int]]) -> dict[int, list[int]]:
    """Return a map from node id to incident cycle ids."""

    membership: dict[int, list[int]] = {}
    for cycle_id, nodes in cycle_nodes.items():
        for node in nodes:
            membership.setdefault(node, []).append(cycle_id)
    return membership


def extract_cycle_ordering_from_metadata(
    G: nx.Graph,
    metadata: dict[str, Any],
) -> dict[int, list[int]]:
    """Recover ordered cycle motif node lists from metadata when available."""

    cycle_nodes = metadata.get("cycle_nodes")
    if isinstance(cycle_nodes, dict):
        return {int(cycle_id): list(nodes) for cycle_id, nodes in cycle_nodes.items()}

    inferred = {}
    for idx, cycle in enumerate(nx.cycle_basis(G)):
        inferred[idx] = list(cycle)
    return inferred


def extract_attachment_map_from_metadata(
    metadata: dict[str, Any],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Map each cycle-tree pair to its attachment metadata."""

    attachments = metadata.get("attachments", [])
    return {
        tuple(sorted(attachment["cycle_pair"])): attachment
        for attachment in attachments
        if "cycle_pair" in attachment
    }


def extract_branch_children_from_metadata(
    metadata: dict[str, Any],
) -> dict[int, list[int]]:
    """Build a children list for each branch-tree parent from metadata."""

    branch_parent_map = dict(metadata.get("branch_parent_map", {}))
    children: dict[int, list[int]] = {}
    for child, parent in branch_parent_map.items():
        children.setdefault(parent, []).append(child)
    for parent, child_list in children.items():
        child_list.sort()
    return children


def compute_backbone_nodes(G: nx.Graph, metadata: dict[str, Any]) -> set[int]:
    """Return the nodes that belong to the pre-branch backbone graph."""

    branch_parent_map = metadata.get("branch_parent_map")
    if isinstance(branch_parent_map, dict):
        branch_nodes_set = {int(node) for node in branch_parent_map}
        return {node for node in G.nodes() if node not in branch_nodes_set}
    return set(G.nodes())


def infer_node_cycle_id(
    node: int,
    node_cycle_membership: dict[int, list[int]],
) -> int | None:
    """Infer a representative cycle id for a node when available."""

    cycle_ids = node_cycle_membership.get(node, [])
    return cycle_ids[0] if len(cycle_ids) == 1 else None


def infer_edge_cycle_id(
    u: int,
    v: int,
    node_cycle_membership: dict[int, list[int]],
) -> int | None:
    """Infer a representative cycle id shared by an edge when available."""

    common = sorted(set(node_cycle_membership.get(u, [])).intersection(node_cycle_membership.get(v, [])))
    return common[0] if len(common) == 1 else None


def infer_edge_branch_depth(
    u: int,
    v: int,
    branch_depth_map: dict[int, int],
) -> int | None:
    """Infer a representative branch depth for an edge-sampled point."""

    depths = [branch_depth_map[node] for node in (u, v) if node in branch_depth_map]
    return max(depths) if depths else None


def infer_edge_branch_root(
    u: int,
    v: int,
    branch_root_map: dict[int, int],
) -> int | None:
    """Infer a representative branch root for an edge-sampled point."""

    roots = {branch_root_map[node] for node in (u, v) if node in branch_root_map}
    if len(roots) == 1:
        return next(iter(roots))
    return None
