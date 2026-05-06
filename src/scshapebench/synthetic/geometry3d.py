"""3D geometry and constructive placement helpers."""

from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np

from .geometry2d import PlacementError, polygon_radius_for_cycle

__all__ = [
    "PlacementError",
    "candidate_branch_clearance_3d",
    "choose_outward_direction_3d",
    "place_cycle_polygon_3d",
    "place_cycle_polygon_from_anchor_3d",
    "place_polyline_between_anchors_3d",
    "place_polyline_path_3d",
    "point_segment_distance_3d",
    "random_unit_vector_3d",
    "segment_distance_3d",
    "spread_child_directions_3d",
    "validate_embedding_3d",
    "validate_partial_embedding_3d",
]


_LOCAL_NEIGHBORHOOD_CACHE_3D: dict[tuple[int, int, int, int], dict[int, set[int]]] = {}


def place_cycle_polygon_3d(
    cycle_nodes: list[int],
    edge_lengths: list[float],
    center: np.ndarray,
    normal: np.ndarray,
    rotation: float,
) -> dict[int, np.ndarray]:
    """Place a cycle as a regular polygon inside a plane in 3D."""

    radius = polygon_radius_for_cycle(edge_lengths)
    u_axis, v_axis = _basis_for_plane(normal)
    positions: dict[int, np.ndarray] = {}
    m = len(cycle_nodes)
    for idx, node in enumerate(cycle_nodes):
        angle = rotation + (2.0 * math.pi * idx / m)
        positions[node] = center + radius * (
            math.cos(angle) * u_axis + math.sin(angle) * v_axis
        )
    return positions


def place_cycle_polygon_from_anchor_3d(
    cycle_nodes: list[int],
    edge_lengths: list[float],
    anchor_node: int,
    anchor_position: np.ndarray,
    center: np.ndarray,
    plane_normal: np.ndarray,
) -> dict[int, np.ndarray]:
    """Place a cycle polygon in 3D with one anchor vertex already fixed."""

    if anchor_node not in cycle_nodes:
        raise ValueError(f"Anchor node {anchor_node} is not in the cycle.")

    anchor_index = cycle_nodes.index(anchor_node)
    u_axis, v_axis = _basis_for_plane(plane_normal)
    anchor_vector = anchor_position - center
    if np.linalg.norm(anchor_vector) < 1e-12:
        rotation = 0.0
    else:
        rotation = math.atan2(
            float(np.dot(anchor_vector, v_axis)),
            float(np.dot(anchor_vector, u_axis)),
        )
        rotation -= 2.0 * math.pi * anchor_index / len(cycle_nodes)
    return place_cycle_polygon_3d(
        cycle_nodes=cycle_nodes,
        edge_lengths=edge_lengths,
        center=center,
        normal=plane_normal,
        rotation=rotation,
    )


def place_polyline_path_3d(
    path_nodes: list[int],
    edge_lengths: list[float],
    start_position: np.ndarray,
    direction: np.ndarray,
    rng: random.Random,
    angular_jitter: float = 0.12,
) -> dict[int, np.ndarray]:
    """Place a path extending outward from one known endpoint in 3D."""

    if len(path_nodes) < 2:
        return {}

    positions: dict[int, np.ndarray] = {}
    current = start_position.copy()
    current_direction = _normalize(direction)

    for idx, edge_length in enumerate(edge_lengths):
        jitter_axis = _random_orthogonal_unit(current_direction, rng)
        tangent_jitter = _random_orthogonal_unit(jitter_axis, rng)
        current_direction = _normalize(
            current_direction
            + math.tan(rng.uniform(-angular_jitter, angular_jitter)) * jitter_axis
            + math.tan(rng.uniform(-angular_jitter, angular_jitter)) * tangent_jitter
        )
        current = current + current_direction * edge_length
        positions[path_nodes[idx + 1]] = current.copy()

    return positions


def place_polyline_between_anchors_3d(
    path_nodes: list[int],
    edge_lengths: list[float],
    start_position: np.ndarray,
    end_position: np.ndarray,
    preferred_normal: np.ndarray,
    feature_scale: float,
    rng: random.Random,
) -> dict[int, np.ndarray]:
    """Place a polyline path between two known anchor points in 3D."""

    if len(path_nodes) <= 2:
        return {}

    chord = end_position - start_position
    chord_length = float(np.linalg.norm(chord))
    tangent = _normalize(chord if chord_length > 1e-12 else random_unit_vector_3d(rng))
    normal = preferred_normal - float(np.dot(preferred_normal, tangent)) * tangent
    if np.linalg.norm(normal) < 1e-10:
        normal = _random_orthogonal_unit(tangent, rng)
    normal = _normalize(normal)

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
        high = feature_scale
        while polyline_total_length(positions_for_height(high)) < total_path_length:
            high *= 1.6
            if high > 25.0 * feature_scale:
                break
        low = 0.0
        for _ in range(30):
            mid = 0.5 * (low + high)
            if polyline_total_length(positions_for_height(mid)) < total_path_length:
                low = mid
            else:
                high = mid
        height = high

    height *= 1.0 + rng.uniform(-0.15, 0.15)
    internal_positions = positions_for_height(height)
    return {
        node: coords.copy() for node, coords in zip(path_nodes[1:-1], internal_positions)
    }


def choose_outward_direction_3d(
    node: int,
    pos: dict[int, np.ndarray],
    cycle_membership: dict[int, list[int]],
    cycle_centers: dict[int, np.ndarray],
    backbone_centroid: np.ndarray,
    incoming_direction: np.ndarray | None,
    rng: random.Random,
) -> np.ndarray:
    """Choose an outward direction for branch growth at a placed node in 3D."""

    if incoming_direction is not None and np.linalg.norm(incoming_direction) > 1e-10:
        return _normalize(incoming_direction)

    membership = cycle_membership.get(node, [])
    if membership:
        local_center = np.mean(
            np.stack([cycle_centers[cycle_id] for cycle_id in membership]),
            axis=0,
        )
        direction = pos[node] - local_center
        if np.linalg.norm(direction) > 1e-10:
            return _normalize(direction)

    direction = pos[node] - backbone_centroid
    if np.linalg.norm(direction) > 1e-10:
        return _normalize(direction)
    return random_unit_vector_3d(rng)


def spread_child_directions_3d(
    preferred_direction: np.ndarray,
    num_children: int,
    spread: float,
    rng: random.Random,
    continuation_child: int | None,
    parent: int,
    branch_depth_map: dict[int, int],
) -> list[np.ndarray]:
    """Spread child branch directions around a preferred direction in 3D."""

    axis = _normalize(preferred_direction)
    if num_children == 1:
        return [axis]

    u_axis, v_axis = _basis_around_axis(axis)
    cone_angle = min(max(spread, 0.2), 1.25)
    azimuth_shift = rng.uniform(0.0, 2.0 * math.pi)
    continuation_first = (
        continuation_child is not None
        and branch_depth_map.get(continuation_child) == branch_depth_map.get(parent)
    )

    directions: list[np.ndarray] = []
    if continuation_first:
        directions.append(axis)
        indices = range(num_children - 1)
    else:
        indices = range(num_children)

    count = max(1, len(list(indices)))
    for idx in indices:
        azimuth = azimuth_shift + 2.0 * math.pi * idx / count
        local_angle = cone_angle * rng.uniform(0.8, 1.05)
        radial = math.cos(azimuth) * u_axis + math.sin(azimuth) * v_axis
        direction = _normalize(
            math.cos(local_angle) * axis + math.sin(local_angle) * radial
        )
        directions.append(direction)

    return directions[:num_children]


def point_segment_distance_3d(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Return the minimum distance from a point to a segment in 3D."""

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab)) / denom
    t = min(1.0, max(0.0, t))
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def segment_distance_3d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> float:
    """Return the minimum Euclidean distance between two 3D segments."""

    u = b - a
    v = d - c
    w = a - c
    a_dot = float(np.dot(u, u))
    b_dot = float(np.dot(u, v))
    c_dot = float(np.dot(v, v))
    d_dot = float(np.dot(u, w))
    e_dot = float(np.dot(v, w))
    denom = a_dot * c_dot - b_dot * b_dot
    small = 1e-12

    if a_dot < small and c_dot < small:
        return float(np.linalg.norm(a - c))
    if a_dot < small:
        return point_segment_distance_3d(a, c, d)
    if c_dot < small:
        return point_segment_distance_3d(c, a, b)

    s_num = 0.0
    s_den = denom
    t_num = 0.0
    t_den = denom

    if denom < small:
        s_num = 0.0
        s_den = 1.0
        t_num = e_dot
        t_den = c_dot
    else:
        s_num = b_dot * e_dot - c_dot * d_dot
        t_num = a_dot * e_dot - b_dot * d_dot

        if s_num < 0.0:
            s_num = 0.0
            t_num = e_dot
            t_den = c_dot
        elif s_num > s_den:
            s_num = s_den
            t_num = e_dot + b_dot
            t_den = c_dot

    if t_num < 0.0:
        t_num = 0.0
        if -d_dot < 0.0:
            s_num = 0.0
        elif -d_dot > a_dot:
            s_num = s_den
        else:
            s_num = -d_dot
            s_den = a_dot
    elif t_num > t_den:
        t_num = t_den
        if (-d_dot + b_dot) < 0.0:
            s_num = 0.0
        elif (-d_dot + b_dot) > a_dot:
            s_num = s_den
        else:
            s_num = -d_dot + b_dot
            s_den = a_dot

    sc = 0.0 if abs(s_num) < small else s_num / s_den
    tc = 0.0 if abs(t_num) < small else t_num / t_den
    delta = w + sc * u - tc * v
    return float(np.linalg.norm(delta))


def validate_embedding_3d(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    target_min_separation: float,
    tolerance: float,
) -> tuple[bool, dict[str, float]]:
    """Validate a 3D embedding against nonadjacent separation constraints."""

    stats = _min_nonadjacent_separation_3d(G, pos)
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


def validate_partial_embedding_3d(
    G: nx.Graph,
    existing_pos: dict[int, np.ndarray],
    candidate_pos: dict[int, np.ndarray],
    target_min_separation: float,
    tolerance_ratio: float = 0.05,
) -> bool:
    """Check whether newly proposed 3D geometry is compatible with existing geometry."""

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
            if point_segment_distance_3d(merged_pos[node], merged_pos[u], merged_pos[v]) < min_allowed:
                return False

    for idx, (u1, v1) in enumerate(candidate_edges):
        for u2, v2 in existing_edges + candidate_edges[idx + 1 :]:
            if len({u1, v1, u2, v2}) < 4:
                continue
            if _edges_locally_related(G, (u1, v1), (u2, v2)):
                continue
            if segment_distance_3d(merged_pos[u1], merged_pos[v1], merged_pos[u2], merged_pos[v2]) < min_allowed:
                return False

    return True


def candidate_branch_clearance_3d(
    G: nx.Graph,
    existing_pos: dict[int, np.ndarray],
    parent: int,
    child: int,
    child_position: np.ndarray,
) -> float:
    """Score the local geometric clearance for a candidate branch edge in 3D."""

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
        clearance = min(
            clearance,
            point_segment_distance_3d(child_position, existing_pos[u], existing_pos[v]),
        )

        if len({parent, child, u, v}) == 4 and not _edges_locally_related(G, (parent, child), (u, v)):
            clearance = min(
                clearance,
                segment_distance_3d(
                    existing_pos[parent],
                    child_position,
                    existing_pos[u],
                    existing_pos[v],
                ),
            )

    return clearance


def random_unit_vector_3d(rng: random.Random) -> np.ndarray:
    """Sample a random unit vector in 3D."""

    z = rng.uniform(-1.0, 1.0)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    radial = math.sqrt(max(0.0, 1.0 - z * z))
    return np.array(
        [radial * math.cos(theta), radial * math.sin(theta), z],
        dtype=float,
    )


def _min_nonadjacent_separation_3d(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
) -> dict[str, float]:
    node_list = list(pos)
    edge_list = [tuple(edge) for edge in G.edges() if edge[0] in pos and edge[1] in pos]

    min_node = math.inf
    for idx, u in enumerate(node_list):
        for v in node_list[idx + 1 :]:
            if _nodes_locally_related(G, u, v):
                continue
            min_node = min(min_node, float(np.linalg.norm(pos[u] - pos[v])))

    min_node_edge = math.inf
    for node in node_list:
        for u, v in edge_list:
            if node in (u, v):
                continue
            if _nodes_locally_related(G, node, u) or _nodes_locally_related(G, node, v):
                continue
            min_node_edge = min(
                min_node_edge,
                point_segment_distance_3d(pos[node], pos[u], pos[v]),
            )

    min_edge = math.inf
    for idx, (u1, v1) in enumerate(edge_list):
        for u2, v2 in edge_list[idx + 1 :]:
            if len({u1, v1, u2, v2}) < 4:
                continue
            if _edges_locally_related(G, (u1, v1), (u2, v2)):
                continue
            min_edge = min(
                min_edge,
                segment_distance_3d(pos[u1], pos[v1], pos[u2], pos[v2]),
            )

    return {
        "min_nonadjacent_node_distance": min_node,
        "min_nonincident_edge_distance": min_edge,
        "min_nonincident_node_edge_distance": min_node_edge,
    }


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vector / norm


def _basis_for_plane(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = _normalize(normal)
    return _basis_around_axis(axis)


def _basis_around_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = _normalize(axis)
    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    u_axis = np.cross(axis, reference)
    if np.linalg.norm(u_axis) < 1e-12:
        reference = np.array([0.0, 0.0, 1.0], dtype=float)
        u_axis = np.cross(axis, reference)
    u_axis = _normalize(u_axis)
    v_axis = _normalize(np.cross(axis, u_axis))
    return u_axis, v_axis


def _random_orthogonal_unit(direction: np.ndarray, rng: random.Random) -> np.ndarray:
    u_axis, v_axis = _basis_around_axis(direction)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    return _normalize(math.cos(angle) * u_axis + math.sin(angle) * v_axis)


def _get_local_neighborhoods(
    G: nx.Graph,
    cutoff: int,
) -> dict[int, set[int]]:
    cache_key = (id(G), G.number_of_nodes(), G.number_of_edges(), cutoff)
    cached = _LOCAL_NEIGHBORHOOD_CACHE_3D.get(cache_key)
    if cached is not None:
        return cached

    neighborhoods = {
        node: set(nx.single_source_shortest_path_length(G, node, cutoff=cutoff))
        for node in G.nodes()
    }
    _LOCAL_NEIGHBORHOOD_CACHE_3D[cache_key] = neighborhoods
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
