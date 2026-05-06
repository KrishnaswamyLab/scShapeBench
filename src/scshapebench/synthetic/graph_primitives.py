"""Basic graph statistics and primitives shared across the pipeline."""

from __future__ import annotations

import networkx as nx

__all__ = [
    "betti_1",
    "branch_nodes",
    "is_planar_graph",
    "leaf_count",
    "planar_layout_positions",
]


def betti_1(G: nx.Graph) -> int:
    """Return the first Betti number of an undirected graph."""

    if G.number_of_nodes() == 0:
        return 0
    return G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)


def leaf_count(G: nx.Graph) -> int:
    """Return the number of degree-1 vertices."""

    return sum(1 for _, degree in G.degree() if degree == 1)


def branch_nodes(G: nx.Graph) -> list[int]:
    """Return graph nodes with degree at least 3."""

    return sorted(node for node, degree in G.degree() if degree >= 3)


def is_planar_graph(G: nx.Graph) -> bool:
    """Return whether the graph is planar."""

    is_planar, _ = nx.check_planarity(G)
    return is_planar


def planar_layout_positions(G: nx.Graph) -> dict[int, tuple[float, float]]:
    """Return a crossing-free planar layout for a planar graph."""

    is_planar, embedding = nx.check_planarity(G)
    if not is_planar:
        raise ValueError("Graph is not planar.")
    return nx.planar_layout(embedding)


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)
