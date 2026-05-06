"""2D geometry and constructive placement helpers."""

from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np

__all__ = [
    "PlacementError",
    "angularly_spread_child_directions",
    "candidate_branch_clearance",
    "choose_outward_direction",
    "min_nonadjacent_separation",
    "order_branch_children",
    "place_cycle_polygon",
    "place_cycle_polygon_from_anchor",
    "place_polyline_between_anchors",
    "place_polyline_path",
    "point_segment_distance_2d",
    "polygon_radius_for_cycle",
    "segment_distance_2d",
    "segments_intersect_2d",
    "validate_embedding",
    "validate_partial_embedding",
]


_LOCAL_NEIGHBORHOOD_CACHE: dict[tuple[int, int, int, int], dict[int, set[int]]] = {}


class PlacementError(RuntimeError):
    """Internal exception used to trigger local/global embedding resampling."""


def polygon_radius_for_cycle(edge_lengths: list[float]) -> float:
    """Estimate a circumradius for a near-regular polygon cycle."""

    if not edge_lengths:
        return 0.0
    m = len(edge_lengths)
    mean_edge = sum(edge_lengths) / m
    return mean_edge / (2.0 * math.sin(math.pi / m))


def place_cycle_polygon(
    cycle_nodes: list[int],
    edge_lengths: list[float],
    center: np.ndarray,
    rotation: float,
) -> dict[int, np.ndarray]:
    """Place a cycle as a regular polygon centered at ``center``."""

    radius = polygon_radius_for_cycle(edge_lengths)
    positions: dict[int, np.ndarray] = {}
    m = len(cycle_nodes)
    for idx, node in enumerate(cycle_nodes):
        angle = rotation + (2.0 * math.pi * idx / m)
        positions[node] = center + radius * np.array([math.cos(angle), math.sin(angle)])
    return positions


def place_cycle_polygon_from_anchor(
    cycle_nodes: list[int],
    edge_lengths: list[float],
    anchor_node: int,
    anchor_position: np.ndarray,
    center: np.ndarray,
) -> dict[int, np.ndarray]:
    """Place a cycle polygon with one specified anchor vertex already fixed."""

    if anchor_node not in cycle_nodes:
        raise ValueError(f"Anchor node {anchor_node} is not in the cycle.")

    anchor_index = cycle_nodes.index(anchor_node)
    anchor_vector = anchor_position - center
    anchor_angle = math.atan2(anchor_vector[1], anchor_vector[0])
    rotation = anchor_angle - (2.0 * math.pi * anchor_index / len(cycle_nodes))
    return place_cycle_polygon(cycle_nodes, edge_lengths, center, rotation)


def place_polyline_path(
    path_nodes: list[int],
    edge_lengths: list[float],
    start_position: np.ndarray,
    direction: np.ndarray,
    rng: random.Random,
    angular_jitter: float = 0.12,
) -> dict[int, np.ndarray]:
    """Place a path extending outward from one known endpoint."""

    if len(path_nodes) < 2:
        return {}

    unit_direction = _normalize(direction)
    positions: dict[int, np.ndarray] = {}
    current = start_position.copy()
    current_direction = unit_direction.copy()

    for idx, edge_length in enumerate(edge_lengths):
        current_direction = _rotate(
            current_direction,
            rng.uniform(-angular_jitter, angular_jitter),
        )
        current = current + current_direction * edge_length
        positions[path_nodes[idx + 1]] = current.copy()

    return positions


def place_polyline_between_anchors(
    path_nodes: list[int],
    edge_lengths: list[float],
    start_position: np.ndarray,
    end_position: np.ndarray,
    preferred_normal: np.ndarray,
    feature_scale: float,
    rng: random.Random,
) -> dict[int, np.ndarray]:
    """Place a polyline path between two already known anchor points."""

    if len(path_nodes) <= 2:
        return {}

    chord = end_position - start_position
    chord_length = float(np.linalg.norm(chord))
    tangent = _normalize(chord if chord_length > 1e-12 else _random_unit_vector(rng))
    normal = np.array([-tangent[1], tangent[0]])
    if float(np.dot(normal, preferred_normal)) < 0.0:
        normal = -normal

    total_path_length = sum(edge_lengths)
    ts = np.cumsum(edge_lengths[:-1]) / total_path_length

    def positions_for_height(height: float) -> list[np.ndarray]:
        return [
            start_position + t * chord + 4.0 * height * t * (1.0 - t) * normal
            for t in ts
        ]

    def polyline_total_length(internal_positions: list[np.ndarray]) -> float:
        points = [start_position, *internal_positions, end_position]
        return sum(
            float(np.linalg.norm(points[idx + 1] - points[idx]))
            for idx in range(len(points) - 1)
        )

    if total_path_length <= chord_length + 1e-9:
        height = 0.0
    else:
        target_length = total_path_length
        high = feature_scale
        while polyline_total_length(positions_for_height(high)) < target_length:
            high *= 1.6
            if high > 25.0 * feature_scale:
                break
        low = 0.0
        for _ in range(30):
            mid = 0.5 * (low + high)
            if polyline_total_length(positions_for_height(mid)) < target_length:
                low = mid
            else:
                high = mid
        height = high

    height *= 1.0 + rng.uniform(-0.15, 0.15)
    internal_positions = positions_for_height(height)
    return {
        node: coords.copy()
        for node, coords in zip(path_nodes[1:-1], internal_positions)
    }


def choose_outward_direction(
    node: int,
    pos: dict[int, np.ndarray],
    cycle_membership: dict[int, list[int]],
    cycle_centers: dict[int, np.ndarray],
    backbone_centroid: np.ndarray,
    incoming_direction: np.ndarray | None,
    rng: random.Random,
) -> np.ndarray:
    """Choose an outward direction for branch growth at a placed node."""

    if incoming_direction is not None and np.linalg.norm(incoming_direction) > 1e-10:
        return _normalize(incoming_direction)

    membership = cycle_membership.get(node, [])
    if membership:
        local_center = np.mean(np.stack([cycle_centers[cycle_id] for cycle_id in membership]), axis=0)
        direction = pos[node] - local_center
        if np.linalg.norm(direction) > 1e-10:
            return _normalize(direction)

    direction = pos[node] - backbone_centroid
    if np.linalg.norm(direction) > 1e-10:
        return _normalize(direction)
    return _random_unit_vector(rng)


def angularly_spread_child_directions(
    preferred_direction: np.ndarray,
    num_children: int,
    spread: float,
    rng: random.Random,
    continuation_child: int | None,
    parent: int,
    branch_depth_map: dict[int, int],
) -> list[np.ndarray]:
    """Spread child branch directions around a preferred outward direction."""

    base_angle = math.atan2(preferred_direction[1], preferred_direction[0])
    if num_children == 1:
        return [_vector_from_angle(base_angle)]

    if continuation_child is not None and branch_depth_map.get(continuation_child) == branch_depth_map.get(parent):
        offsets = [0.0]
        side_count = num_children - 1
        if side_count > 0:
            side_offsets = np.linspace(-spread, spread, side_count + 2)[1:-1]
            offsets.extend(float(offset) for offset in side_offsets)
    else:
        offsets = [float(offset) for offset in np.linspace(-spread, spread, num_children)]

    if len(offsets) > 1:
        tail = offsets[1:]
        rng.shuffle(tail)
        offsets = [offsets[0], *tail]
    return [_vector_from_angle(base_angle + offset) for offset in offsets[:num_children]]


def segment_distance_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> float:
    """Return the minimum Euclidean distance between two 2D line segments."""

    if segments_intersect_2d(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance_2d(a, c, d),
        point_segment_distance_2d(b, c, d),
        point_segment_distance_2d(c, a, b),
        point_segment_distance_2d(d, a, b),
    )


def point_segment_distance_2d(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Return the minimum distance from a point to a segment in 2D."""

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab)) / denom
    t = min(1.0, max(0.0, t))
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def min_nonadjacent_separation(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
) -> dict[str, float]:
    """Measure minimum nonadjacent node, node-edge, and edge-edge separations."""

    node_list = list(pos)
    edge_list = [tuple(edge) for edge in G.edges() if edge[0] in pos and edge[1] in pos]

    min_node = math.inf
    for idx, u in enumerate(node_list):
        for v in node_list[idx + 1 :]:
            if _nodes_locally_related(G, u, v):
                continue
            dist = float(np.linalg.norm(pos[u] - pos[v]))
            min_node = min(min_node, dist)

    min_node_edge = math.inf
    for node in node_list:
        for u, v in edge_list:
            if node in (u, v):
                continue
            if _nodes_locally_related(G, node, u) or _nodes_locally_related(G, node, v):
                continue
            dist = point_segment_distance_2d(pos[node], pos[u], pos[v])
            min_node_edge = min(min_node_edge, dist)

    min_edge = math.inf
    for idx, (u1, v1) in enumerate(edge_list):
        for u2, v2 in edge_list[idx + 1 :]:
            if len({u1, v1, u2, v2}) < 4:
                continue
            if _edges_locally_related(G, (u1, v1), (u2, v2)):
                continue
            dist = segment_distance_2d(pos[u1], pos[v1], pos[u2], pos[v2])
            min_edge = min(min_edge, dist)

    return {
        "min_nonadjacent_node_distance": min_node,
        "min_nonincident_edge_distance": min_edge,
        "min_nonincident_node_edge_distance": min_node_edge,
    }


def validate_embedding(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    target_min_separation: float,
    tolerance: float,
) -> tuple[bool, dict[str, float]]:
    """Validate an embedding against nonadjacent separation constraints."""

    stats = min_nonadjacent_separation(G, pos)
    if len(pos) != G.number_of_nodes():
        return False, stats

    thresholds = [
        stats["min_nonadjacent_node_distance"],
        stats["min_nonincident_edge_distance"],
        stats["min_nonincident_node_edge_distance"],
    ]
    for value in thresholds:
        if math.isfinite(value) and value < target_min_separation - tolerance:
            return False, stats
    return True, stats


def validate_partial_embedding(
    G: nx.Graph,
    existing_pos: dict[int, np.ndarray],
    candidate_pos: dict[int, np.ndarray],
    target_min_separation: float,
    tolerance_ratio: float = 0.05,
) -> bool:
    """Check whether newly proposed geometry is compatible with existing geometry."""

    if not candidate_pos:
        return True

    min_allowed = target_min_separation * (1.0 - tolerance_ratio)
    merged_pos = dict(existing_pos)
    merged_pos.update(candidate_pos)
    candidate_nodes = list(candidate_pos)
    existing_edges = [
        tuple(edge) for edge in G.edges() if edge[0] in existing_pos and edge[1] in existing_pos
    ]
    candidate_edges = [
        tuple(edge)
        for edge in G.edges()
        if edge[0] in merged_pos
        and edge[1] in merged_pos
        and (edge[0] in candidate_pos or edge[1] in candidate_pos)
    ]

    for idx, u in enumerate(candidate_nodes):
        for v in list(merged_pos):
            if u == v:
                continue
            if v in candidate_nodes[:idx]:
                continue
            if _nodes_locally_related(G, u, v):
                continue
            if float(np.linalg.norm(merged_pos[u] - merged_pos[v])) < min_allowed:
                return False

    for node in candidate_nodes:
        for u, v in existing_edges + candidate_edges:
            if node in (u, v):
                continue
            if _nodes_locally_related(G, node, u) or _nodes_locally_related(G, node, v):
                continue
            if point_segment_distance_2d(merged_pos[node], merged_pos[u], merged_pos[v]) < min_allowed:
                return False

    for idx, (u1, v1) in enumerate(candidate_edges):
        for u2, v2 in existing_edges + candidate_edges[idx + 1 :]:
            if len({u1, v1, u2, v2}) < 4:
                continue
            if _edges_locally_related(G, (u1, v1), (u2, v2)):
                continue
            if segment_distance_2d(merged_pos[u1], merged_pos[v1], merged_pos[u2], merged_pos[v2]) < min_allowed:
                return False

    return True


def candidate_branch_clearance(
    G: nx.Graph,
    existing_pos: dict[int, np.ndarray],
    parent: int,
    child: int,
    child_position: np.ndarray,
) -> float:
    """Score the local geometric clearance for a candidate branch edge."""

    clearance = math.inf
    for node, coords in existing_pos.items():
        if _nodes_locally_related(G, child, node):
            continue
        clearance = min(clearance, float(np.linalg.norm(child_position - coords)))

    for u, v in G.edges():
        if u not in existing_pos or v not in existing_pos:
            continue
        if _nodes_locally_related(G, child, u) or _nodes_locally_related(G, child, v):
            continue
        clearance = min(clearance, point_segment_distance_2d(child_position, existing_pos[u], existing_pos[v]))

        if len({parent, child, u, v}) == 4 and not _edges_locally_related(G, (parent, child), (u, v)):
            clearance = min(
                clearance,
                segment_distance_2d(existing_pos[parent], child_position, existing_pos[u], existing_pos[v]),
            )

    return clearance


def order_branch_children(
    children: list[int],
    parent: int,
    branch_depth_map: dict[int, int],
) -> list[int]:
    """Order branch children so path continuation is considered first when present."""

    parent_depth = branch_depth_map.get(parent)
    continuation = [
        child for child in children if parent_depth is not None and branch_depth_map.get(child) == parent_depth
    ]
    other_children = [child for child in children if child not in continuation]
    continuation.sort()
    other_children.sort()
    return continuation + other_children


def segments_intersect_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    """Return whether two line segments intersect in 2D."""

    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    if o1 == 0.0 and on_segment(a, c, b):
        return True
    if o2 == 0.0 and on_segment(a, d, b):
        return True
    if o3 == 0.0 and on_segment(c, a, d):
        return True
    if o4 == 0.0 and on_segment(c, b, d):
        return True
    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return vector / norm


def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def _vector_from_angle(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)], dtype=float)


def _perpendicular(vector: np.ndarray) -> np.ndarray:
    return np.array([-vector[1], vector[0]], dtype=float)


def _random_unit_vector(rng: random.Random) -> np.ndarray:
    return _vector_from_angle(rng.uniform(0.0, 2.0 * math.pi))


def _get_local_neighborhoods(
    G: nx.Graph,
    cutoff: int,
) -> dict[int, set[int]]:
    cache_key = (id(G), G.number_of_nodes(), G.number_of_edges(), cutoff)
    cached = _LOCAL_NEIGHBORHOOD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    neighborhoods = {
        node: set(nx.single_source_shortest_path_length(G, node, cutoff=cutoff))
        for node in G.nodes()
    }
    _LOCAL_NEIGHBORHOOD_CACHE[cache_key] = neighborhoods
    return neighborhoods


def _nodes_locally_related(G: nx.Graph, u: int, v: int) -> bool:
    return v in _get_local_neighborhoods(G, cutoff=3).get(u, set())


def _edges_locally_related(
    G: nx.Graph,
    edge_a: tuple[int, int],
    edge_b: tuple[int, int],
) -> bool:
    for u in edge_a:
        for v in edge_b:
            if _nodes_locally_related(G, u, v):
                return True
    return False


def _perpendicular_unit(vector: np.ndarray) -> np.ndarray:
    return _perpendicular(_normalize(vector))
