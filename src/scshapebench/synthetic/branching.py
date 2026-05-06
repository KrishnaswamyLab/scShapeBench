"""Tree-branch augmentation for an existing backbone graph."""

from __future__ import annotations

import random
from typing import Any

import networkx as nx

from .graph_primitives import betti_1, branch_nodes, is_planar_graph, leaf_count

__all__ = [
    "add_random_branches",
    "attach_path_branch",
    "get_next_node_id",
    "grow_branches_from_nodes",
    "promote_branch_nodes",
    "sample_attachment_nodes",
    "validate_branching_does_not_change_cycles",
]


def add_random_branches(
    G: nx.Graph,
    branch_attach_count_range: tuple[int, int] = (1, 4),
    children_per_branch_point_range: tuple[int, int] = (1, 3),
    branch_length_range: tuple[int, int] = (1, 4),
    max_branch_depth_range: tuple[int, int] = (1, 2),
    promotion_prob_range: tuple[float, float] = (0.2, 0.6),
    max_degree: int = 5,
    max_total_nodes: int | None = None,
    max_total_edges: int | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Attach random tree-like branches to an existing connected graph."""

    _validate_branching_inputs(
        branch_attach_count_range=branch_attach_count_range,
        children_per_branch_point_range=children_per_branch_point_range,
        branch_length_range=branch_length_range,
        max_branch_depth_range=max_branch_depth_range,
        promotion_prob_range=promotion_prob_range,
        max_degree=max_degree,
        max_total_nodes=max_total_nodes,
        max_total_edges=max_total_edges,
    )
    if not nx.is_connected(G):
        raise ValueError("Input graph G must be connected before branching.")
    _validate_existing_graph_size(
        G=G,
        max_total_nodes=max_total_nodes,
        max_total_edges=max_total_edges,
    )

    rng = random.Random(seed)
    betti_before = betti_1(G)
    next_node_id = get_next_node_id(G)

    num_initial_attachment_sites = rng.randint(
        branch_attach_count_range[0], branch_attach_count_range[1]
    )
    max_branch_depth = rng.randint(max_branch_depth_range[0], max_branch_depth_range[1])
    promotion_prob = rng.uniform(promotion_prob_range[0], promotion_prob_range[1])

    sampled_branch_params = {
        "num_initial_attachment_sites": num_initial_attachment_sites,
        "max_branch_depth": max_branch_depth,
        "promotion_prob": promotion_prob,
        "children_per_branch_point_range": children_per_branch_point_range,
        "branch_length_range": branch_length_range,
        "max_degree": max_degree,
        "max_total_nodes": max_total_nodes,
        "max_total_edges": max_total_edges,
    }

    initial_attachment_nodes = sample_attachment_nodes(
        G=G,
        k=num_initial_attachment_sites,
        max_degree=max_degree,
        rng=rng,
    )

    branch_generation_nodes: dict[int, list[int]] = {
        generation: [] for generation in range(1, max_branch_depth + 1)
    }
    branch_parent_map: dict[int, int] = {}
    branch_root_map: dict[int, int] = {}
    branch_depth_map: dict[int, int] = {}

    current_parents = list(initial_attachment_nodes)
    current_parent_roots = {node: node for node in current_parents}

    for generation in range(1, max_branch_depth + 1):
        if not current_parents:
            break

        new_nodes, next_node_id = grow_branches_from_nodes(
            G=G,
            parent_nodes=current_parents,
            parent_root_lookup=current_parent_roots,
            generation=generation,
            children_per_branch_point_range=children_per_branch_point_range,
            branch_length_range=branch_length_range,
            max_degree=max_degree,
            max_total_nodes=max_total_nodes,
            max_total_edges=max_total_edges,
            next_node_id=next_node_id,
            rng=rng,
        )

        branch_generation_nodes[generation] = list(new_nodes["generation_nodes"])
        branch_parent_map.update(new_nodes["parent_map"])
        branch_root_map.update(new_nodes["root_map"])
        branch_depth_map.update(new_nodes["depth_map"])

        if generation == max_branch_depth:
            break

        promoted_nodes = promote_branch_nodes(
            G=G,
            candidate_nodes=branch_generation_nodes[generation],
            promotion_prob=promotion_prob,
            max_degree=max_degree,
            rng=rng,
        )
        current_parents = promoted_nodes
        current_parent_roots = {
            node: branch_root_map[node] for node in promoted_nodes if node in branch_root_map
        }

    validate_branching_does_not_change_cycles(G, betti_before)

    output_metadata = dict(metadata) if metadata is not None else {}
    output_metadata["branch_params"] = sampled_branch_params
    output_metadata["branch_attachment_nodes_initial"] = list(initial_attachment_nodes)
    output_metadata["branch_generation_nodes"] = branch_generation_nodes
    output_metadata["branch_parent_map"] = branch_parent_map
    output_metadata["branch_root_map"] = branch_root_map
    output_metadata["branch_depth_map"] = branch_depth_map
    output_metadata["num_leaves_after_branching"] = leaf_count(G)
    output_metadata["num_branch_nodes_after_branching"] = len(branch_nodes(G))
    output_metadata["betti_1_after_branching"] = betti_1(G)

    output_metadata["leaf_count"] = leaf_count(G)
    output_metadata["branch_nodes"] = branch_nodes(G)
    output_metadata["betti_1"] = betti_1(G)
    output_metadata["is_planar"] = is_planar_graph(G)

    return G, output_metadata


def _validate_branching_inputs(
    branch_attach_count_range: tuple[int, int],
    children_per_branch_point_range: tuple[int, int],
    branch_length_range: tuple[int, int],
    max_branch_depth_range: tuple[int, int],
    promotion_prob_range: tuple[float, float],
    max_degree: int,
    max_total_nodes: int | None,
    max_total_edges: int | None,
) -> None:
    if branch_attach_count_range[0] > branch_attach_count_range[1]:
        raise ValueError(f"Invalid branch_attach_count_range {branch_attach_count_range}.")
    if branch_attach_count_range[0] < 0:
        raise ValueError("branch_attach_count_range must be non-negative.")

    if children_per_branch_point_range[0] > children_per_branch_point_range[1]:
        raise ValueError(
            f"Invalid children_per_branch_point_range {children_per_branch_point_range}."
        )
    if children_per_branch_point_range[0] < 1:
        raise ValueError("children_per_branch_point_range must have minimum at least 1.")

    if branch_length_range[0] > branch_length_range[1]:
        raise ValueError(f"Invalid branch_length_range {branch_length_range}.")
    if branch_length_range[0] < 1:
        raise ValueError("branch_length_range must have minimum at least 1.")

    if max_branch_depth_range[0] > max_branch_depth_range[1]:
        raise ValueError(f"Invalid max_branch_depth_range {max_branch_depth_range}.")
    if max_branch_depth_range[0] < 1:
        raise ValueError("max_branch_depth_range must have minimum at least 1.")

    if promotion_prob_range[0] > promotion_prob_range[1]:
        raise ValueError(f"Invalid promotion_prob_range {promotion_prob_range}.")
    if promotion_prob_range[0] < 0.0 or promotion_prob_range[1] > 1.0:
        raise ValueError("promotion_prob_range must lie inside [0.0, 1.0].")

    if max_degree < 1:
        raise ValueError("max_degree must be at least 1.")
    if max_total_nodes is not None and max_total_nodes < 1:
        raise ValueError("max_total_nodes must be at least 1 when provided.")
    if max_total_edges is not None and max_total_edges < 0:
        raise ValueError("max_total_edges must be non-negative when provided.")


def _validate_existing_graph_size(
    G: nx.Graph,
    max_total_nodes: int | None,
    max_total_edges: int | None,
) -> None:
    if max_total_nodes is not None and G.number_of_nodes() > max_total_nodes:
        raise ValueError(
            f"Input graph already exceeds max_total_nodes={max_total_nodes}: "
            f"{G.number_of_nodes()} nodes."
        )
    if max_total_edges is not None and G.number_of_edges() > max_total_edges:
        raise ValueError(
            f"Input graph already exceeds max_total_edges={max_total_edges}: "
            f"{G.number_of_edges()} edges."
        )


def get_next_node_id(G: nx.Graph) -> int:
    """Return the next fresh integer node id for a graph with integer labels."""

    if G.number_of_nodes() == 0:
        return 0
    if not all(isinstance(node, int) for node in G.nodes()):
        raise ValueError("Branch generation requires integer node ids.")
    return max(G.nodes()) + 1


def sample_attachment_nodes(
    G: nx.Graph,
    k: int,
    max_degree: int,
    rng: random.Random,
) -> list[int]:
    """Sample up to ``k`` eligible attachment nodes from the current graph."""

    eligible = [node for node, degree in G.degree() if degree < max_degree]
    if not eligible or k <= 0:
        return []
    if len(eligible) <= k:
        rng.shuffle(eligible)
        return eligible
    return rng.sample(eligible, k)


def attach_path_branch(
    G: nx.Graph,
    parent: int,
    path_length: int,
    next_node_id: int,
) -> tuple[list[int], int]:
    """Attach a fresh path of ``path_length`` edges to ``parent``."""

    if path_length < 1:
        raise ValueError(f"path_length must be at least 1, got {path_length}.")

    new_nodes = list(range(next_node_id, next_node_id + path_length))
    if not new_nodes:
        return [], next_node_id

    G.add_edge(parent, new_nodes[0])
    for u, v in zip(new_nodes, new_nodes[1:]):
        G.add_edge(u, v)
    return new_nodes, next_node_id + path_length


def grow_branches_from_nodes(
    G: nx.Graph,
    parent_nodes: list[int],
    parent_root_lookup: dict[int, int],
    generation: int,
    children_per_branch_point_range: tuple[int, int],
    branch_length_range: tuple[int, int],
    max_degree: int,
    max_total_nodes: int | None,
    max_total_edges: int | None,
    next_node_id: int,
    rng: random.Random,
) -> tuple[dict[str, Any], int]:
    """Grow one branching generation from the provided parent nodes."""

    generation_nodes: list[int] = []
    parent_map: dict[int, int] = {}
    root_map: dict[int, int] = {}
    depth_map: dict[int, int] = {}

    for parent in parent_nodes:
        remaining_capacity = max_degree - G.degree(parent)
        if remaining_capacity <= 0:
            continue

        sampled_children = rng.randint(
            children_per_branch_point_range[0],
            children_per_branch_point_range[1],
        )
        num_children = min(sampled_children, remaining_capacity)
        if num_children <= 0:
            continue

        root_node = parent_root_lookup[parent]
        for _ in range(num_children):
            path_length = rng.randint(branch_length_range[0], branch_length_range[1])
            _validate_projected_branch_size(
                G=G,
                path_length=path_length,
                max_total_nodes=max_total_nodes,
                max_total_edges=max_total_edges,
            )
            new_path_nodes, next_node_id = attach_path_branch(
                G=G,
                parent=parent,
                path_length=path_length,
                next_node_id=next_node_id,
            )
            generation_nodes.extend(new_path_nodes)

            previous = parent
            for node in new_path_nodes:
                parent_map[node] = previous
                root_map[node] = root_node
                depth_map[node] = generation
                previous = node

    return {
        "generation_nodes": generation_nodes,
        "parent_map": parent_map,
        "root_map": root_map,
        "depth_map": depth_map,
    }, next_node_id


def _validate_projected_branch_size(
    G: nx.Graph,
    path_length: int,
    max_total_nodes: int | None,
    max_total_edges: int | None,
) -> None:
    projected_nodes = G.number_of_nodes() + path_length
    projected_edges = G.number_of_edges() + path_length
    if max_total_nodes is not None and projected_nodes > max_total_nodes:
        raise ValueError(
            f"Branch growth would exceed max_total_nodes={max_total_nodes}: "
            f"{projected_nodes} projected nodes."
        )
    if max_total_edges is not None and projected_edges > max_total_edges:
        raise ValueError(
            f"Branch growth would exceed max_total_edges={max_total_edges}: "
            f"{projected_edges} projected edges."
        )


def promote_branch_nodes(
    G: nx.Graph,
    candidate_nodes: list[int],
    promotion_prob: float,
    max_degree: int,
    rng: random.Random,
) -> list[int]:
    """Promote a subset of newly created nodes to be branch points."""

    promoted: list[int] = []
    for node in candidate_nodes:
        if G.degree(node) >= max_degree:
            continue
        if rng.random() < promotion_prob:
            promoted.append(node)
    return promoted


def validate_branching_does_not_change_cycles(G: nx.Graph, betti_before: int) -> None:
    """Assert that branching preserved connectivity and cycle count."""

    if not nx.is_connected(G):
        raise RuntimeError("Branch growth disconnected the graph.")
    if betti_1(G) != betti_before:
        raise RuntimeError(
            f"Branch growth changed Betti-1 from {betti_before} to {betti_1(G)}."
        )
