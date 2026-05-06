"""Cycle-backbone topology generation."""

from __future__ import annotations

from collections import deque
import heapq
import random
from typing import Any

import networkx as nx

from .graph_primitives import betti_1, branch_nodes, is_planar_graph, leaf_count
from .metadata_utils import cycle_edge_membership

__all__ = [
    "attach_cycles_by_path",
    "attach_cycles_by_shared_path",
    "attach_cycles_by_shared_vertex",
    "generate_cycle_backbone",
    "make_cycle",
    "random_tree_on_cycles",
]


def make_cycle(length: int, next_node_id: int) -> tuple[nx.Graph, list[int], int]:
    """Create a simple cycle graph with fresh global integer node ids."""

    if length < 3:
        raise ValueError(f"Cycle length must be at least 3, got {length}.")

    nodes = list(range(next_node_id, next_node_id + length))
    subgraph = nx.Graph()
    subgraph.add_nodes_from(nodes)
    for idx, node in enumerate(nodes):
        subgraph.add_edge(node, nodes[(idx + 1) % length])
    return subgraph, nodes, next_node_id + length


def attach_cycles_by_path(
    G: nx.Graph,
    cyc_a_nodes: list[int],
    cyc_b_nodes: list[int],
    path_len: int,
    next_node_id: int,
    rng: random.Random,
) -> tuple[int, dict[str, Any]]:
    """Connect two disjoint cycles by a fresh path of exactly ``path_len`` edges."""

    if path_len <= 0:
        raise ValueError(f"path_len must be positive, got {path_len}.")

    a_anchor = rng.choice(cyc_a_nodes)
    b_anchor = rng.choice(cyc_b_nodes)

    if path_len == 1:
        G.add_edge(a_anchor, b_anchor)
        path_nodes = [a_anchor, b_anchor]
        return next_node_id, {
            "attachment_type": "path",
            "anchor_a": a_anchor,
            "anchor_b": b_anchor,
            "path_nodes": path_nodes,
        }

    interior = list(range(next_node_id, next_node_id + path_len - 1))
    path_nodes = [a_anchor, *interior, b_anchor]
    for u, v in zip(path_nodes, path_nodes[1:]):
        G.add_edge(u, v)
    return next_node_id + path_len - 1, {
        "attachment_type": "path",
        "anchor_a": a_anchor,
        "anchor_b": b_anchor,
        "path_nodes": path_nodes,
    }


def attach_cycles_by_shared_vertex(
    G: nx.Graph,
    cyc_a_nodes: list[int],
    cyc_b_nodes: list[int],
    rng: random.Random,
) -> tuple[list[int], dict[str, Any]]:
    """Glue two cycles by identifying one randomly chosen vertex from each."""

    a_anchor = rng.choice(cyc_a_nodes)
    b_anchor = rng.choice(cyc_b_nodes)
    merged_b_nodes = _replace_nodes(cyc_b_nodes, {b_anchor: a_anchor})
    _identify_nodes(G, keep=a_anchor, remove=b_anchor)
    return merged_b_nodes, {
        "attachment_type": "shared_vertex",
        "anchor_a": a_anchor,
        "anchor_b_original": b_anchor,
        "shared_node": a_anchor,
    }


def attach_cycles_by_shared_path(
    G: nx.Graph,
    cyc_a_nodes: list[int],
    cyc_b_nodes: list[int],
    shared_edges: int,
    cycle_a_start_idx: int,
    cycle_b_start_idx: int,
    rng: random.Random,
) -> tuple[list[int], dict[str, Any]]:
    """Glue two cycles along a shared contiguous path of ``shared_edges`` edges."""

    if shared_edges <= 0:
        raise ValueError(f"shared_edges must be positive, got {shared_edges}.")
    if shared_edges >= len(cyc_a_nodes) - 1 or shared_edges >= len(cyc_b_nodes) - 1:
        raise ValueError(
            "Shared path must be strictly shorter than both cycles minus one edge."
        )

    a_path = _path_from_cycle_start(cyc_a_nodes, cycle_a_start_idx, shared_edges)
    b_path = _path_from_cycle_start(cyc_b_nodes, cycle_b_start_idx, shared_edges)
    a_edge_indices = {
        (cycle_a_start_idx + offset) % len(cyc_a_nodes) for offset in range(shared_edges)
    }
    b_edge_indices = {
        (cycle_b_start_idx + offset) % len(cyc_b_nodes) for offset in range(shared_edges)
    }
    same_orientation = rng.choice([True, False])
    if not same_orientation:
        b_path = list(reversed(b_path))

    mapping = {b_node: a_node for a_node, b_node in zip(a_path, b_path)}
    merged_b_nodes = _replace_nodes(cyc_b_nodes, mapping)
    for b_node, a_node in mapping.items():
        _identify_nodes(G, keep=a_node, remove=b_node)

    return merged_b_nodes, {
        "attachment_type": "shared_path",
        "shared_edges": shared_edges,
        "cycle_a_path": a_path,
        "cycle_b_path_original": b_path,
        "orientation": "same" if same_orientation else "reversed",
        "shared_path_nodes": a_path,
        "cycle_a_edge_indices": sorted(a_edge_indices),
        "cycle_b_edge_indices": sorted(b_edge_indices),
    }


def random_tree_on_cycles(num_cycles: int, rng: random.Random) -> list[tuple[int, int]]:
    """Sample a uniform random labeled tree on ``0, 1, ..., num_cycles - 1``."""

    if num_cycles < 0:
        raise ValueError(f"num_cycles must be non-negative, got {num_cycles}.")
    if num_cycles <= 1:
        return []
    if num_cycles == 2:
        return [(0, 1)]

    prufer = [rng.randrange(num_cycles) for _ in range(num_cycles - 2)]
    degree = [1] * num_cycles
    for node in prufer:
        degree[node] += 1

    leaves = [node for node, deg in enumerate(degree) if deg == 1]
    heapq.heapify(leaves)
    edges: list[tuple[int, int]] = []

    for node in prufer:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, node))
        degree[leaf] -= 1
        degree[node] -= 1
        if degree[node] == 1:
            heapq.heappush(leaves, node)

    a = heapq.heappop(leaves)
    b = heapq.heappop(leaves)
    edges.append((a, b))
    return edges


def generate_cycle_backbone(
    num_cycles: int,
    cycle_len_range: tuple[int, int] = (4, 8),
    sep_range: tuple[int, int] = (-2, 3),
    seed: int | None = None,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Generate a connected graph consisting of cycle motifs or a singleton root."""

    _validate_inputs(num_cycles, cycle_len_range, sep_range)
    if num_cycles == 0:
        G = nx.Graph()
        G.add_node(0)
        metadata: dict[str, Any] = {
            "cycle_tree_edges": [],
            "cycle_lengths": {},
            "pair_separations": {},
            "cycle_nodes": {},
            "cycle_motifs": {},
            "attachments": [],
            "edge_cycle_membership": {},
            "num_cycles_ground_truth": 0,
            "betti_1": 0,
            "leaf_count": leaf_count(G),
            "branch_nodes": branch_nodes(G),
            "is_planar": True,
            "max_edge_cycle_multiplicity": 0,
        }
        return G, metadata

    rng = random.Random(seed)
    cycle_tree_edges = [tuple(sorted(edge)) for edge in random_tree_on_cycles(num_cycles, rng)]
    tree_adj = _tree_adjacency(num_cycles, cycle_tree_edges)
    cycle_lengths = _sample_feasible_cycle_lengths(
        num_cycles=num_cycles,
        cycle_len_range=cycle_len_range,
        sep_range=sep_range,
        tree_edges=cycle_tree_edges,
        rng=rng,
    )
    pair_separations = _sample_pair_separations(
        cycle_lengths=cycle_lengths,
        sep_range=sep_range,
        tree_edges=cycle_tree_edges,
        rng=rng,
    )
    shared_path_allocations = _allocate_shared_path_segments(
        cycle_lengths=cycle_lengths,
        pair_separations=pair_separations,
        rng=rng,
    )

    G = nx.Graph()
    cycle_nodes: dict[int, list[int]] = {}
    cycle_motifs: dict[int, dict[str, Any]] = {}
    attachments: list[dict[str, Any]] = []

    next_node_id = 0
    root = 0
    root_graph, root_nodes, next_node_id = make_cycle(cycle_lengths[root], next_node_id)
    G.add_nodes_from(root_graph.nodes())
    G.add_edges_from(root_graph.edges())
    cycle_nodes[root] = list(root_nodes)
    cycle_motifs[root] = {
        "cycle_id": root,
        "original_cycle_length": cycle_lengths[root],
        "original_motif_vertices": list(root_nodes),
        "final_nodes": list(root_nodes),
    }

    parent: dict[int, int | None] = {root: None}
    traversal = deque([root])
    while traversal:
        current = traversal.popleft()
        for neighbor in tree_adj[current]:
            if neighbor in parent:
                continue
            parent[neighbor] = current
            traversal.append(neighbor)

            child_graph, child_nodes, next_node_id = make_cycle(
                cycle_lengths[neighbor], next_node_id
            )
            original_child_nodes = list(child_nodes)
            G.add_nodes_from(child_graph.nodes())
            G.add_edges_from(child_graph.edges())

            pair = tuple(sorted((current, neighbor)))
            separation = pair_separations[pair]

            if separation > 0:
                next_node_id, attachment_info = attach_cycles_by_path(
                    G=G,
                    cyc_a_nodes=cycle_nodes[current],
                    cyc_b_nodes=child_nodes,
                    path_len=separation,
                    next_node_id=next_node_id,
                    rng=rng,
                )
            elif separation == 0:
                child_nodes, attachment_info = attach_cycles_by_shared_vertex(
                    G=G,
                    cyc_a_nodes=cycle_nodes[current],
                    cyc_b_nodes=child_nodes,
                    rng=rng,
                )
            else:
                child_nodes, attachment_info = attach_cycles_by_shared_path(
                    G=G,
                    cyc_a_nodes=cycle_nodes[current],
                    cyc_b_nodes=child_nodes,
                    shared_edges=-separation,
                    cycle_a_start_idx=shared_path_allocations[(current, pair)]["start_idx"],
                    cycle_b_start_idx=shared_path_allocations[(neighbor, pair)]["start_idx"],
                    rng=rng,
                )

            cycle_nodes[neighbor] = list(child_nodes)
            cycle_motifs[neighbor] = {
                "cycle_id": neighbor,
                "original_cycle_length": cycle_lengths[neighbor],
                "original_motif_vertices": original_child_nodes,
                "final_nodes": list(child_nodes),
            }
            attachments.append(
                {
                    "cycle_pair": pair,
                    "separation": separation,
                    **attachment_info,
                }
            )

    edge_membership = cycle_edge_membership(cycle_nodes)
    max_edge_multiplicity = max((len(cycle_ids) for cycle_ids in edge_membership.values()), default=0)

    metadata: dict[str, Any] = {
        "cycle_tree_edges": cycle_tree_edges,
        "cycle_lengths": dict(cycle_lengths),
        "pair_separations": dict(pair_separations),
        "cycle_nodes": {cycle_id: list(nodes) for cycle_id, nodes in cycle_nodes.items()},
        "cycle_motifs": cycle_motifs,
        "attachments": attachments,
        "edge_cycle_membership": edge_membership,
        "num_cycles_ground_truth": num_cycles,
        "betti_1": betti_1(G),
        "leaf_count": leaf_count(G),
        "branch_nodes": branch_nodes(G),
        "is_planar": None,
        "max_edge_cycle_multiplicity": max_edge_multiplicity,
    }

    if not nx.is_connected(G):
        raise RuntimeError("Internal error: generated graph is not connected.")
    if metadata["betti_1"] != num_cycles:
        raise RuntimeError(
            f"Internal error: expected Betti-1 {num_cycles}, got {metadata['betti_1']}."
        )
    if not is_planar_graph(G):
        raise RuntimeError("Internal error: generated graph is not planar.")
    if metadata["max_edge_cycle_multiplicity"] > 2:
        raise RuntimeError(
            "Internal error: a final graph edge belongs to more than two cycle motifs."
        )

    metadata["is_planar"] = True
    return G, metadata


def _validate_inputs(
    num_cycles: int,
    cycle_len_range: tuple[int, int],
    sep_range: tuple[int, int],
) -> None:
    if num_cycles < 0:
        raise ValueError(f"num_cycles must be non-negative, got {num_cycles}.")

    min_len, max_len = cycle_len_range
    if min_len > max_len:
        raise ValueError(f"Invalid cycle_len_range {cycle_len_range}.")
    if min_len < 3:
        raise ValueError("Cycle lengths must be at least 3.")

    sep_min, sep_max = sep_range
    if sep_min > sep_max:
        raise ValueError(f"Invalid sep_range {sep_range}.")

    if num_cycles > 0 and not _assignment_feasible(cycle_len_range, sep_range):
        raise ValueError(
            "cycle_len_range and sep_range are incompatible: no valid pairwise "
            "cycle attachment is possible."
        )


def _assignment_feasible(
    cycle_len_range: tuple[int, int],
    sep_range: tuple[int, int],
) -> bool:
    min_len, max_len = cycle_len_range
    for len_a in range(min_len, max_len + 1):
        for len_b in range(min_len, max_len + 1):
            if _valid_separations_for_pair(len_a, len_b, sep_range):
                return True
    return False


def _sample_feasible_cycle_lengths(
    num_cycles: int,
    cycle_len_range: tuple[int, int],
    sep_range: tuple[int, int],
    tree_edges: list[tuple[int, int]],
    rng: random.Random,
    max_attempts: int = 5000,
) -> dict[int, int]:
    min_len, max_len = cycle_len_range

    for _ in range(max_attempts):
        lengths = {
            cycle_id: rng.randint(min_len, max_len) for cycle_id in range(num_cycles)
        }
        if all(
            _valid_separations_for_pair(lengths[i], lengths[j], sep_range)
            for i, j in tree_edges
        ):
            return lengths

    raise ValueError(
        "Unable to sample cycle lengths compatible with sep_range after repeated attempts."
    )


def _sample_pair_separations(
    cycle_lengths: dict[int, int],
    sep_range: tuple[int, int],
    tree_edges: list[tuple[int, int]],
    rng: random.Random,
    max_attempts: int = 5000,
) -> dict[tuple[int, int], int]:
    for _ in range(max_attempts):
        pair_separations: dict[tuple[int, int], int] = {}
        for i, j in tree_edges:
            valid = _valid_separations_for_pair(cycle_lengths[i], cycle_lengths[j], sep_range)
            if not valid:
                raise RuntimeError("Internal error: no valid separation for a sampled tree edge.")
            pair_separations[(i, j)] = rng.choice(valid)

        if _shared_path_budget_feasible(cycle_lengths, pair_separations):
            return pair_separations

    raise ValueError(
        "Unable to sample separations without reusing cycle edges across shared-path overlaps."
    )


def _valid_separations_for_pair(
    len_a: int,
    len_b: int,
    sep_range: tuple[int, int],
) -> list[int]:
    valid: list[int] = []
    sep_min, sep_max = sep_range
    for sep in range(sep_min, sep_max + 1):
        if sep >= 0:
            valid.append(sep)
            continue

        shared_edges = -sep
        if shared_edges < len_a - 1 and shared_edges < len_b - 1:
            valid.append(sep)
    return valid


def _shared_path_budget_feasible(
    cycle_lengths: dict[int, int],
    pair_separations: dict[tuple[int, int], int],
) -> bool:
    used_overlap_edges = {cycle_id: 0 for cycle_id in cycle_lengths}
    for (i, j), separation in pair_separations.items():
        if separation >= 0:
            continue
        shared_edges = -separation
        used_overlap_edges[i] += shared_edges
        used_overlap_edges[j] += shared_edges

    return all(used_overlap_edges[cycle_id] <= cycle_lengths[cycle_id] for cycle_id in cycle_lengths)


def _allocate_shared_path_segments(
    cycle_lengths: dict[int, int],
    pair_separations: dict[tuple[int, int], int],
    rng: random.Random,
) -> dict[tuple[int, tuple[int, int]], dict[str, Any]]:
    allocations: dict[tuple[int, tuple[int, int]], dict[str, Any]] = {}
    requests_by_cycle: dict[int, list[tuple[tuple[int, int], int]]] = {
        cycle_id: [] for cycle_id in cycle_lengths
    }

    for pair, separation in pair_separations.items():
        if separation >= 0:
            continue
        shared_edges = -separation
        i, j = pair
        requests_by_cycle[i].append((pair, shared_edges))
        requests_by_cycle[j].append((pair, shared_edges))

    for cycle_id, requests in requests_by_cycle.items():
        if not requests:
            continue

        requests = list(requests)
        rng.shuffle(requests)
        cycle_len = cycle_lengths[cycle_id]
        total_shared_edges = sum(shared_edges for _, shared_edges in requests)
        slack = cycle_len - total_shared_edges
        gaps = _random_cyclic_gaps(slack, len(requests), rng)
        cursor = rng.randrange(cycle_len)

        for gap, (pair, shared_edges) in zip(gaps, requests):
            cursor = (cursor + gap) % cycle_len
            start_idx = cursor
            edge_indices = {
                (start_idx + offset) % cycle_len for offset in range(shared_edges)
            }
            allocations[(cycle_id, pair)] = {
                "start_idx": start_idx,
                "edge_indices": sorted(edge_indices),
            }
            cursor = (cursor + shared_edges) % cycle_len

    return allocations


def _tree_adjacency(
    num_cycles: int,
    tree_edges: list[tuple[int, int]],
) -> dict[int, list[int]]:
    adj = {cycle_id: [] for cycle_id in range(num_cycles)}
    for i, j in tree_edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _random_cyclic_gaps(total_gap: int, num_gaps: int, rng: random.Random) -> list[int]:
    if num_gaps <= 0:
        return []
    if num_gaps == 1:
        return [total_gap]

    gaps: list[int] = []
    remaining_gap = total_gap
    for _ in range(num_gaps - 1):
        gap = rng.randint(0, remaining_gap)
        gaps.append(gap)
        remaining_gap -= gap
    gaps.append(remaining_gap)
    rng.shuffle(gaps)
    return gaps


def _path_from_cycle_start(
    cyc_nodes: list[int],
    start_idx: int,
    shared_edges: int,
) -> list[int]:
    return [cyc_nodes[(start_idx + offset) % len(cyc_nodes)] for offset in range(shared_edges + 1)]


def _replace_nodes(nodes: list[int], mapping: dict[int, int]) -> list[int]:
    return [mapping.get(node, node) for node in nodes]


def _identify_nodes(G: nx.Graph, keep: int, remove: int) -> None:
    if keep == remove:
        return
    if remove not in G:
        return

    for neighbor in list(G.neighbors(remove)):
        if neighbor != keep:
            G.add_edge(keep, neighbor)
    G.remove_node(remove)
