"""Dimension-agnostic constructive geometry helpers."""

from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np

__all__ = [
    "candidate_branch_clearance_nd",
    "choose_outward_direction_nd",
    "inflate_complement_positions_nd",
    "normalize_nd",
    "place_cycle_polygon_from_anchor_nd",
    "place_cycle_polygon_nd",
    "place_polyline_between_anchors_nd",
    "place_polyline_path_nd",
    "point_segment_distance_nd",
    "random_orthogonal_unit",
    "random_unit_vector_nd",
    "sample_branch_direction_nd",
    "segment_distance_nd",
    "spread_child_directions_nd",
    "validate_embedding_nd",
    "validate_partial_embedding_nd",
]


_LOCAL_NEIGHBORHOOD_CACHE_ND: dict[tuple[int, int, int, int], dict[int, set[int]]] = {}


def normalize_nd(vector: np.ndarray) -> np.ndarray:
    """Return a normalized vector, falling back to the first basis axis."""

    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm >= 1e-12:
        return vector / norm

    fallback = np.zeros(vector.shape[0], dtype=float)
    fallback[0] = 1.0
    return fallback


def random_unit_vector_nd(dimension: int, rng: random.Random) -> np.ndarray:
    """Sample a random unit vector in ``R^dimension``."""

    if dimension < 1:
        raise ValueError("dimension must be positive.")

    sample = np.array([rng.gauss(0.0, 1.0) for _ in range(dimension)], dtype=float)
    if float(np.linalg.norm(sample)) < 1e-12:
        sample[0] = 1.0
    return normalize_nd(sample)


def random_orthogonal_unit(direction: np.ndarray, rng: random.Random) -> np.ndarray:
    """Sample a random unit vector orthogonal to ``direction``."""

    axis = normalize_nd(direction)
    dimension = axis.shape[0]

    for _ in range(16):
        candidate = random_unit_vector_nd(dimension, rng)
        candidate = candidate - float(np.dot(candidate, axis)) * axis
        norm = float(np.linalg.norm(candidate))
        if norm >= 1e-12:
            return candidate / norm

    basis_index = int(np.argmin(np.abs(axis)))
    candidate = np.zeros(dimension, dtype=float)
    candidate[basis_index] = 1.0
    candidate = candidate - float(np.dot(candidate, axis)) * axis
    return normalize_nd(candidate)


def place_cycle_polygon_nd(
    cycle_nodes: list[int],
    edge_lengths: list[float],
    center: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    rotation: float,
) -> dict[int, np.ndarray]:
    """Place a cycle as a regular polygon inside a 2-plane in ``R^d``."""

    if not edge_lengths:
        return {}

    radius = _polygon_radius_for_cycle(edge_lengths)
    u_axis = normalize_nd(u_axis)
    v_axis = _project_orthogonal(v_axis, [u_axis])
    if float(np.linalg.norm(v_axis)) < 1e-12:
        raise ValueError("Expected u_axis and v_axis to span a plane.")
    v_axis = normalize_nd(v_axis)

    positions: dict[int, np.ndarray] = {}
    m = len(cycle_nodes)
    for idx, node in enumerate(cycle_nodes):
        angle = rotation + (2.0 * math.pi * idx / m)
        positions[node] = center + radius * (
            math.cos(angle) * u_axis + math.sin(angle) * v_axis
        )
    return positions


def place_cycle_polygon_from_anchor_nd(
    cycle_nodes: list[int],
    edge_lengths: list[float],
    anchor_node: int,
    anchor_position: np.ndarray,
    center: np.ndarray,
    rng: random.Random,
    v_axis_hint: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Place a cycle polygon in ``R^d`` while keeping one anchor fixed."""

    if anchor_node not in cycle_nodes:
        raise ValueError(f"Anchor node {anchor_node} is not in the cycle.")

    anchor_index = cycle_nodes.index(anchor_node)
    u_axis = anchor_position - center
    if float(np.linalg.norm(u_axis)) < 1e-12:
        u_axis = random_unit_vector_nd(center.shape[0], rng)
    u_axis = normalize_nd(u_axis)

    if v_axis_hint is None:
        v_axis = random_orthogonal_unit(u_axis, rng)
    else:
        v_axis = _project_orthogonal(np.asarray(v_axis_hint, dtype=float), [u_axis])
        if float(np.linalg.norm(v_axis)) < 1e-12:
            v_axis = random_orthogonal_unit(u_axis, rng)
        else:
            v_axis = normalize_nd(v_axis)

    rotation = -(2.0 * math.pi * anchor_index / len(cycle_nodes))
    return place_cycle_polygon_nd(
        cycle_nodes=cycle_nodes,
        edge_lengths=edge_lengths,
        center=center,
        u_axis=u_axis,
        v_axis=v_axis,
        rotation=rotation,
    )


def place_polyline_path_nd(
    path_nodes: list[int],
    edge_lengths: list[float],
    start_position: np.ndarray,
    direction: np.ndarray,
    rng: random.Random,
    angular_jitter: float = 0.12,
) -> dict[int, np.ndarray]:
    """Place a path extending outward from one known endpoint in ``R^d``."""

    if len(path_nodes) < 2:
        return {}

    positions: dict[int, np.ndarray] = {}
    current = np.asarray(start_position, dtype=float).copy()
    current_direction = normalize_nd(direction)
    dimension = current_direction.shape[0]

    for idx, edge_length in enumerate(edge_lengths):
        if angular_jitter > 0.0:
            if dimension == 2:
                angle = math.atan2(current_direction[1], current_direction[0])
                angle += rng.uniform(-angular_jitter, angular_jitter)
                current_direction = np.array(
                    [math.cos(angle), math.sin(angle)],
                    dtype=float,
                )
            else:
                perturbation = random_orthogonal_unit(current_direction, rng)
                current_direction = normalize_nd(
                    current_direction
                    + math.tan(rng.uniform(-angular_jitter, angular_jitter)) * perturbation
                )
        current = current + current_direction * edge_length
        positions[path_nodes[idx + 1]] = current.copy()

    return positions


def place_polyline_between_anchors_nd(
    path_nodes: list[int],
    edge_lengths: list[float],
    start_position: np.ndarray,
    end_position: np.ndarray,
    preferred_normal: np.ndarray,
    feature_scale: float,
    rng: random.Random,
) -> dict[int, np.ndarray]:
    """Place a polyline path between two known anchor points in ``R^d``."""

    if len(path_nodes) <= 2:
        return {}

    chord = np.asarray(end_position, dtype=float) - np.asarray(start_position, dtype=float)
    chord_length = float(np.linalg.norm(chord))
    tangent = normalize_nd(
        chord if chord_length > 1e-12 else random_unit_vector_nd(chord.shape[0], rng)
    )
    normal = _project_orthogonal(np.asarray(preferred_normal, dtype=float), [tangent])
    if float(np.linalg.norm(normal)) < 1e-12:
        normal = random_orthogonal_unit(tangent, rng)
    else:
        normal = normalize_nd(normal)

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
        node: coords.copy()
        for node, coords in zip(path_nodes[1:-1], internal_positions, strict=False)
    }


def choose_outward_direction_nd(
    node: int,
    pos: dict[int, np.ndarray],
    cycle_membership: dict[int, list[int]],
    cycle_centers: dict[int, np.ndarray],
    backbone_centroid: np.ndarray,
    incoming_direction: np.ndarray | None,
    rng: random.Random,
) -> np.ndarray:
    """Choose an outward branch-growth direction in ``R^d``."""

    if incoming_direction is not None and float(np.linalg.norm(incoming_direction)) > 1e-10:
        return normalize_nd(incoming_direction)

    membership = cycle_membership.get(node, [])
    if membership:
        local_center = np.mean(
            np.stack([cycle_centers[cycle_id] for cycle_id in membership]),
            axis=0,
        )
        direction = pos[node] - local_center
        if float(np.linalg.norm(direction)) > 1e-10:
            return normalize_nd(direction)

    direction = pos[node] - backbone_centroid
    if float(np.linalg.norm(direction)) > 1e-10:
        return normalize_nd(direction)
    return random_unit_vector_nd(backbone_centroid.shape[0], rng)


def spread_child_directions_nd(
    preferred_direction: np.ndarray,
    num_children: int,
    spread: float,
    rng: random.Random,
    continuation_child: int | None,
    parent: int,
    branch_depth_map: dict[int, int],
) -> list[np.ndarray]:
    """Spread child branch directions around a preferred direction."""

    axis = normalize_nd(preferred_direction)
    if num_children == 1:
        return [axis]

    dimension = axis.shape[0]
    continuation_first = (
        continuation_child is not None
        and branch_depth_map.get(continuation_child) == branch_depth_map.get(parent)
    )

    if dimension == 2:
        base_angle = math.atan2(axis[1], axis[0])
        if continuation_first:
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
        return [
            np.array([math.cos(base_angle + offset), math.sin(base_angle + offset)], dtype=float)
            for offset in offsets[:num_children]
        ]

    u_axis = random_orthogonal_unit(axis, rng)
    v_seed = random_unit_vector_nd(dimension, rng)
    v_axis = _project_orthogonal(v_seed, [axis, u_axis])
    if float(np.linalg.norm(v_axis)) < 1e-12:
        v_axis = random_orthogonal_unit(axis + 0.25 * u_axis, rng)
        v_axis = _project_orthogonal(v_axis, [axis, u_axis])
    v_axis = normalize_nd(v_axis)

    cone_angle = min(max(spread, 0.2), 1.25)
    azimuth_shift = rng.uniform(0.0, 2.0 * math.pi)
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
        direction = normalize_nd(
            math.cos(local_angle) * axis + math.sin(local_angle) * radial
        )
        directions.append(direction)

    return directions[:num_children]


def sample_branch_direction_nd(
    base_direction: np.ndarray,
    angle_budget: float,
    rng: random.Random,
) -> np.ndarray:
    """Sample a perturbed branch direction around a base direction."""

    direction = normalize_nd(base_direction)
    if direction.shape[0] == 2:
        angle = math.atan2(direction[1], direction[0])
        angle += rng.uniform(-angle_budget, angle_budget)
        return np.array([math.cos(angle), math.sin(angle)], dtype=float)

    perturbation = random_orthogonal_unit(direction, rng)
    candidate = direction + math.tan(rng.uniform(-angle_budget, angle_budget)) * perturbation
    if float(np.linalg.norm(candidate)) < 1e-10:
        candidate = direction
    return normalize_nd(candidate)


def point_segment_distance_nd(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Return the minimum distance from a point to a segment in ``R^d``."""

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab)) / denom
    t = min(1.0, max(0.0, t))
    projection = a + t * ab
    return float(np.linalg.norm(point - projection))


def segment_distance_nd(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> float:
    """Return the minimum Euclidean distance between two segments in ``R^d``."""

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
        return point_segment_distance_nd(a, c, d)
    if c_dot < small:
        return point_segment_distance_nd(c, a, b)

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


def validate_embedding_nd(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    target_min_separation: float,
    tolerance: float,
) -> tuple[bool, dict[str, float]]:
    """Validate an embedding against nonadjacent separation constraints."""

    stats = _min_nonadjacent_separation_nd(G, pos)
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


def validate_partial_embedding_nd(
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
            if point_segment_distance_nd(merged_pos[node], merged_pos[u], merged_pos[v]) < min_allowed:
                return False

    for idx, (u1, v1) in enumerate(candidate_edges):
        for u2, v2 in existing_edges + candidate_edges[idx + 1 :]:
            if len({u1, v1, u2, v2}) < 4:
                continue
            if merged_pos[u1].shape[0] == 2 and _segments_intersect_2d(
                merged_pos[u1],
                merged_pos[v1],
                merged_pos[u2],
                merged_pos[v2],
            ):
                return False
            if _edges_locally_related(G, (u1, v1), (u2, v2)):
                continue
            if segment_distance_nd(merged_pos[u1], merged_pos[v1], merged_pos[u2], merged_pos[v2]) < min_allowed:
                return False

    return True


def candidate_branch_clearance_nd(
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
        clearance = min(
            clearance,
            point_segment_distance_nd(child_position, existing_pos[u], existing_pos[v]),
        )

        if len({parent, child, u, v}) == 4 and not _edges_locally_related(G, (parent, child), (u, v)):
            clearance = min(
                clearance,
                segment_distance_nd(
                    existing_pos[parent],
                    child_position,
                    existing_pos[u],
                    existing_pos[v],
                ),
            )

    return clearance


def inflate_complement_positions_nd(
    complement_positions: dict[int, np.ndarray],
    complement_path: list[int],
    preferred_normal: np.ndarray,
    edge_seq: list[float],
    target_packing_distance: float,
) -> None:
    """Bow complementary path vertices outward to realize the requested packing."""

    if len(complement_path) <= 2 or not edge_seq:
        return

    normal = normalize_nd(preferred_normal)
    total_length = sum(edge_seq)
    if total_length <= 1e-12:
        return

    cumulative = np.cumsum(edge_seq[:-1], dtype=float)
    for node, distance_along_path in zip(complement_path[1:-1], cumulative, strict=False):
        t = float(distance_along_path / total_length)
        bow = 4.0 * t * (1.0 - t)
        complement_positions[node] = complement_positions[node] + normal * (
            0.35 * target_packing_distance * bow
        )


def _polygon_radius_for_cycle(edge_lengths: list[float]) -> float:
    if not edge_lengths:
        return 0.0
    m = len(edge_lengths)
    mean_edge = sum(edge_lengths) / m
    return mean_edge / (2.0 * math.sin(math.pi / m))


def _project_orthogonal(vector: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    projected = np.asarray(vector, dtype=float).copy()
    for basis_vector in basis:
        projected = projected - float(np.dot(projected, basis_vector)) * basis_vector
    return projected


def _segments_intersect_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    """Return whether two non-incident 2D segments intersect."""

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


def _min_nonadjacent_separation_nd(
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
                point_segment_distance_nd(pos[node], pos[u], pos[v]),
            )

    min_edge = math.inf
    for idx, (u1, v1) in enumerate(edge_list):
        for u2, v2 in edge_list[idx + 1 :]:
            if len({u1, v1, u2, v2}) < 4:
                continue
            if pos[u1].shape[0] == 2 and _segments_intersect_2d(
                pos[u1],
                pos[v1],
                pos[u2],
                pos[v2],
            ):
                min_edge = 0.0
                continue
            if _edges_locally_related(G, (u1, v1), (u2, v2)):
                continue
            min_edge = min(
                min_edge,
                segment_distance_nd(pos[u1], pos[v1], pos[u2], pos[v2]),
            )

    return {
        "min_nonadjacent_node_distance": min_node,
        "min_nonincident_edge_distance": min_edge,
        "min_nonincident_node_edge_distance": min_node_edge,
    }


def _get_local_neighborhoods(
    G: nx.Graph,
    cutoff: int,
) -> dict[int, set[int]]:
    cache_key = (id(G), G.number_of_nodes(), G.number_of_edges(), cutoff)
    cached = _LOCAL_NEIGHBORHOOD_CACHE_ND.get(cache_key)
    if cached is not None:
        return cached

    neighborhoods = {
        node: set(nx.single_source_shortest_path_length(G, node, cutoff=cutoff))
        for node in G.nodes()
    }
    _LOCAL_NEIGHBORHOOD_CACHE_ND[cache_key] = neighborhoods
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
