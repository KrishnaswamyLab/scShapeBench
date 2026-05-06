"""Point-cloud sampling from an embedded graph skeleton."""

from __future__ import annotations

import random
from typing import Any

import networkx as nx
import numpy as np

from .metadata_utils import (
    build_node_cycle_membership,
    extract_cycle_ordering_from_metadata,
    infer_edge_branch_depth,
    infer_edge_branch_root,
    infer_edge_cycle_id,
    infer_node_cycle_id,
)

__all__ = [
    "add_gaussian_noise",
    "get_embedding_params_from_metadata",
    "resolve_positions",
    "resolve_positions_2d",
    "sample_edge_point_count",
    "sample_noisy_point_cloud_from_embedding",
    "sample_points_on_segment",
    "validate_sampled_point_cloud",
]


def sample_noisy_point_cloud_from_embedding(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    metadata: dict[str, Any] | None = None,
    include_node_points: bool = True,
    min_points_per_edge: int = 2,
    density_jitter_ratio_range: tuple[float, float] = (0.0, 0.15),
    endpoint_sampling_mode: str = "include",
    feature_thickness_ratio: float = 0.0,
    tube_sampling_mode: str = "centerline",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample a clean and noisy point cloud from an embedded graph skeleton."""

    if endpoint_sampling_mode not in {"include", "exclude", "half-open"}:
        raise ValueError(
            "endpoint_sampling_mode must be one of 'include', 'exclude', or 'half-open'."
        )
    if min_points_per_edge < 1:
        raise ValueError("min_points_per_edge must be at least 1.")
    if density_jitter_ratio_range[0] > density_jitter_ratio_range[1]:
        raise ValueError(
            f"Invalid density_jitter_ratio_range {density_jitter_ratio_range}."
        )
    if density_jitter_ratio_range[0] < 0.0:
        raise ValueError("density_jitter_ratio_range must be non-negative.")
    if feature_thickness_ratio < 0.0:
        raise ValueError("feature_thickness_ratio must be non-negative.")
    if tube_sampling_mode not in {"centerline", "solid", "surface"}:
        raise ValueError(
            "tube_sampling_mode must be one of 'centerline', 'solid', or 'surface'."
        )

    output_metadata = dict(metadata) if metadata is not None else {}
    embedding_params = get_embedding_params_from_metadata(output_metadata)
    resolved_pos = resolve_positions(G=G, pos=pos, metadata=output_metadata)
    ambient_dimension = (
        next(iter(resolved_pos.values())).shape[0] if resolved_pos else int(output_metadata.get("embedding_dimension", 2))
    )

    feature_scale = embedding_params["feature_scale"]
    density = embedding_params["sampling_density_per_feature"]
    noise_ratio = embedding_params["noise_ratio"]
    sigma = noise_ratio * feature_scale
    feature_thickness_radius = float(feature_thickness_ratio) * float(feature_scale)
    if feature_thickness_radius <= 0.0:
        tube_sampling_mode = "centerline"

    rng = random.Random(seed)
    tube_rng = np.random.default_rng(rng.randrange(1 << 63))
    noise_rng = np.random.default_rng(rng.randrange(1 << 63))

    clean_point_list: list[np.ndarray] = []
    centerline_point_list: list[np.ndarray] = []
    tube_offset_list: list[np.ndarray] = []
    point_source_type: list[str] = []
    point_source_edge: list[list[int] | None] = []
    point_source_node: list[int | None] = []
    point_source_t: list[float | None] = []

    node_cycle_membership = build_node_cycle_membership(
        extract_cycle_ordering_from_metadata(G, output_metadata)
    )
    branch_depth_map = dict(output_metadata.get("branch_depth_map", {}))
    branch_root_map = dict(output_metadata.get("branch_root_map", {}))
    point_branch_depth: list[int | None] = []
    point_branch_root: list[int | None] = []
    point_cycle_id: list[int | None] = []

    num_edge_sampled_points = 0
    for u, v in sorted(G.edges()):
        p0 = resolved_pos[u]
        p1 = resolved_pos[v]
        length = float(np.linalg.norm(p1 - p0))
        n_samples = sample_edge_point_count(
            length=length,
            feature_scale=feature_scale,
            density=density,
            feature_thickness_radius=feature_thickness_radius,
            tube_sampling_mode=tube_sampling_mode,
            density_jitter_ratio_range=density_jitter_ratio_range,
            min_points_per_edge=min_points_per_edge,
            rng=rng,
        )
        edge_points, t_values = sample_points_on_segment(
            p0=p0,
            p1=p1,
            n_samples=n_samples,
            endpoint_sampling_mode=endpoint_sampling_mode,
        )
        edge_offsets = sample_tube_offsets_for_edge(
            p0=p0,
            p1=p1,
            n_samples=edge_points.shape[0],
            radius=feature_thickness_radius,
            mode=tube_sampling_mode,
            rng=tube_rng,
        )
        edge_clean_points = edge_points + edge_offsets

        shared_cycle = infer_edge_cycle_id(u, v, node_cycle_membership)
        shared_branch_depth = infer_edge_branch_depth(u, v, branch_depth_map)
        shared_branch_root = infer_edge_branch_root(u, v, branch_root_map)

        for centerline_point, offset, point, t in zip(
            edge_points, edge_offsets, edge_clean_points, t_values
        ):
            clean_point_list.append(point)
            centerline_point_list.append(centerline_point)
            tube_offset_list.append(offset)
            point_source_type.append("edge")
            point_source_edge.append([u, v])
            point_source_node.append(None)
            point_source_t.append(t)
            point_branch_depth.append(shared_branch_depth)
            point_branch_root.append(shared_branch_root)
            point_cycle_id.append(shared_cycle)
        num_edge_sampled_points += len(t_values)

    num_node_sampled_points = 0
    include_junction_points = include_node_points or G.number_of_edges() == 0
    if include_junction_points:
        for node in sorted(G.nodes()):
            centerline_point = resolved_pos[node].copy()
            if G.degree(node) == 0 and feature_thickness_radius > 0.0:
                n_node_samples = max(
                    sample_node_tube_point_count(
                        radius=feature_thickness_radius,
                        feature_scale=feature_scale,
                        density=density,
                        mode=tube_sampling_mode,
                        dimension=ambient_dimension,
                        include_node_points=True,
                    ),
                    20,
                )
                offsets = sample_tube_offsets_for_node(
                    dimension=ambient_dimension,
                    n_samples=n_node_samples,
                    radius=feature_thickness_radius,
                    mode=tube_sampling_mode,
                    rng=tube_rng,
                )
            else:
                n_node_samples = 1 if include_node_points else 0
                offsets = np.zeros((n_node_samples, ambient_dimension), dtype=float)
            for offset in offsets:
                clean_point_list.append(centerline_point + offset)
                centerline_point_list.append(centerline_point)
                tube_offset_list.append(offset)
                point_source_type.append("node")
                point_source_edge.append(None)
                point_source_node.append(node)
                point_source_t.append(None)
                point_branch_depth.append(branch_depth_map.get(node))
                point_branch_root.append(branch_root_map.get(node))
                point_cycle_id.append(infer_node_cycle_id(node, node_cycle_membership))
            num_node_sampled_points += n_node_samples

    if clean_point_list:
        clean_points = np.vstack(clean_point_list).astype(float, copy=False)
    else:
        clean_points = np.empty((0, ambient_dimension), dtype=float)
    if centerline_point_list:
        centerline_points = np.vstack(centerline_point_list).astype(float, copy=False)
        tube_offsets = np.vstack(tube_offset_list).astype(float, copy=False)
    else:
        centerline_points = np.empty((0, ambient_dimension), dtype=float)
        tube_offsets = np.empty((0, ambient_dimension), dtype=float)

    noisy_points, noise_vectors = add_gaussian_noise(
        points=clean_points,
        sigma=sigma,
        rng=noise_rng,
    )

    validate_sampled_point_cloud(
        G=G,
        clean_points=clean_points,
        noisy_points=noisy_points,
        sigma=sigma,
        include_node_points=include_node_points,
        num_edge_sampled_points=num_edge_sampled_points,
    )

    output_metadata["point_cloud_params"] = {
        "feature_scale": feature_scale,
        "sampling_density_per_feature": density,
        "noise_ratio": noise_ratio,
        "sigma": sigma,
        "embedding_dimension": ambient_dimension,
        "include_node_points": include_node_points,
        "include_junction_points": include_junction_points,
        "node_tube_sampling_policy": "edge_tubes_only_except_isolated_nodes",
        "min_points_per_edge": min_points_per_edge,
        "density_jitter_ratio_range": density_jitter_ratio_range,
        "endpoint_sampling_mode": endpoint_sampling_mode,
        "feature_thickness_ratio": float(feature_thickness_ratio),
        "feature_thickness_radius": float(feature_thickness_radius),
        "tube_sampling_mode": tube_sampling_mode,
    }
    output_metadata["clean_points"] = clean_points.tolist()
    output_metadata["noisy_points"] = noisy_points.tolist()
    output_metadata["point_centerline_points"] = centerline_points.tolist()
    output_metadata["point_tube_offsets"] = tube_offsets.tolist()
    output_metadata["point_source_type"] = point_source_type
    output_metadata["point_source_edge"] = point_source_edge
    output_metadata["point_source_node"] = point_source_node
    output_metadata["point_source_t"] = point_source_t
    output_metadata["point_noise_vectors"] = noise_vectors.tolist()
    output_metadata["num_clean_points"] = int(clean_points.shape[0])
    output_metadata["num_edge_sampled_points"] = num_edge_sampled_points
    output_metadata["num_node_sampled_points"] = num_node_sampled_points
    output_metadata["point_branch_depth"] = point_branch_depth
    output_metadata["point_branch_root"] = point_branch_root
    output_metadata["point_cycle_id"] = point_cycle_id

    return clean_points, noisy_points, output_metadata


def get_embedding_params_from_metadata(metadata: dict[str, Any]) -> dict[str, float]:
    """Extract required embedding difficulty parameters from metadata."""

    embedding_params = metadata.get("embedding_params")
    if not isinstance(embedding_params, dict):
        raise ValueError(
            "Missing embedding_params metadata. Run embed_graph_constructively first."
        )

    required = (
        "feature_scale",
        "sampling_density_per_feature",
        "noise_ratio",
    )
    missing = [key for key in required if key not in embedding_params]
    if missing:
        raise ValueError(
            f"embedding_params is missing required keys: {', '.join(missing)}."
        )

    return {
        "feature_scale": float(embedding_params["feature_scale"]),
        "sampling_density_per_feature": float(
            embedding_params["sampling_density_per_feature"]
        ),
        "noise_ratio": float(embedding_params["noise_ratio"]),
    }


def resolve_positions(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    metadata: dict[str, Any],
) -> dict[int, np.ndarray]:
    """Resolve node positions from the explicit input or serialized metadata."""

    resolved = {node: np.asarray(coords, dtype=float) for node, coords in pos.items()}
    missing = [node for node in G.nodes() if node not in resolved]
    if not missing:
        return resolved

    serialized_candidates = [
        metadata.get("node_positions"),
        metadata.get("node_positions_3d"),
        metadata.get("node_positions_2d"),
    ]
    embedding_dimension = metadata.get("embedding_dimension")
    if embedding_dimension in {2, 3}:
        serialized_candidates.insert(1, metadata.get(f"node_positions_{embedding_dimension}d"))

    for serialized in serialized_candidates:
        if not isinstance(serialized, dict):
            continue
        for node in missing:
            if node in serialized:
                resolved[node] = np.asarray(serialized[node], dtype=float)
            elif str(node) in serialized:
                resolved[node] = np.asarray(serialized[str(node)], dtype=float)
        missing = [node for node in G.nodes() if node not in resolved]
        if not missing:
            break

    missing_after = [node for node in G.nodes() if node not in resolved]
    if missing_after:
        raise ValueError(
            "Missing positions for graph nodes: "
            + ", ".join(str(node) for node in missing_after[:10])
        )
    return resolved


def resolve_positions_2d(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    metadata: dict[str, Any],
) -> dict[int, np.ndarray]:
    """Legacy wrapper that resolves serialized positions without forcing 2D."""

    return resolve_positions(G=G, pos=pos, metadata=metadata)


def sample_edge_point_count(
    length: float,
    feature_scale: float,
    density: float,
    feature_thickness_radius: float,
    tube_sampling_mode: str,
    density_jitter_ratio_range: tuple[float, float],
    min_points_per_edge: int,
    rng: random.Random,
) -> int:
    """Sample the total number of points to place on an edge segment."""

    if feature_scale <= 0.0:
        raise ValueError("feature_scale must be positive.")
    if density <= 0.0:
        raise ValueError("sampling_density_per_feature must be positive.")

    expected_samples = density * (max(length, 0.0) / feature_scale)
    if feature_thickness_radius > 0.0 and tube_sampling_mode != "centerline":
        radius = float(feature_thickness_radius)
        if tube_sampling_mode == "surface":
            thickness_multiplier = 1.0 + (2.0 * np.pi * radius / feature_scale)
        elif tube_sampling_mode == "solid":
            thickness_multiplier = 1.0 + (np.pi * radius * radius / (feature_scale * feature_scale))
        else:
            raise ValueError(f"Unsupported tube sampling mode {tube_sampling_mode!r}.")
        expected_samples *= max(1.0, thickness_multiplier)

    jitter_mag = rng.uniform(
        density_jitter_ratio_range[0],
        density_jitter_ratio_range[1],
    )
    jitter = jitter_mag if rng.random() < 0.5 else -jitter_mag
    adjusted = expected_samples * (1.0 + jitter)
    return max(min_points_per_edge, int(round(adjusted)))


def sample_node_tube_point_count(
    radius: float,
    feature_scale: float,
    density: float,
    mode: str,
    dimension: int,
    include_node_points: bool,
) -> int:
    """Choose enough node samples for tube junctions to visually connect."""

    if radius <= 0.0 or mode == "centerline":
        return 1 if include_node_points else 0

    circumference = 2.0 * np.pi * float(radius)
    expected = float(density) * circumference / max(float(feature_scale), np.finfo(float).eps)

    if mode == "surface":
        minimum = 14 if int(dimension) == 2 else 18
        maximum = 72 if int(dimension) == 2 else 96
    elif mode == "solid":
        minimum = 8 if int(dimension) == 2 else 12
        maximum = 48 if int(dimension) == 2 else 64
    else:
        raise ValueError(f"Unsupported tube sampling mode {mode!r}.")

    return int(np.clip(np.ceil(expected), minimum, maximum))


def sample_points_on_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    n_samples: int,
    endpoint_sampling_mode: str,
) -> tuple[np.ndarray, list[float]]:
    """Sample evenly spaced points on a segment."""

    if n_samples <= 0:
        return np.empty((0, p0.shape[0]), dtype=float), []

    if endpoint_sampling_mode == "include":
        if n_samples == 1:
            t_values = np.array([0.0], dtype=float)
        else:
            t_values = np.linspace(0.0, 1.0, n_samples, dtype=float)
    elif endpoint_sampling_mode == "exclude":
        t_values = np.linspace(0.0, 1.0, n_samples + 2, dtype=float)[1:-1]
    elif endpoint_sampling_mode == "half-open":
        t_values = np.linspace(0.0, 1.0, n_samples, endpoint=False, dtype=float)
    else:
        raise ValueError(f"Unsupported endpoint_sampling_mode {endpoint_sampling_mode!r}.")

    points = (1.0 - t_values)[:, None] * p0[None, :] + t_values[:, None] * p1[None, :]
    return points, [float(t) for t in t_values]


def sample_tube_offsets_for_edge(
    p0: np.ndarray,
    p1: np.ndarray,
    n_samples: int,
    radius: float,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample offsets in the cross-section orthogonal to an edge."""

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    dimension = int(p0.shape[0])
    if n_samples <= 0:
        return np.empty((0, dimension), dtype=float)
    if radius <= 0.0 or mode == "centerline":
        return np.zeros((n_samples, dimension), dtype=float)
    direction = p1 - p0
    basis = _orthonormal_complement(direction)
    local_offsets = _sample_ball_or_sphere(
        n_samples=n_samples,
        dimension=basis.shape[1],
        radius=radius,
        mode=mode,
        rng=rng,
    )
    return local_offsets @ basis.T


def sample_tube_offsets_for_node(
    dimension: int,
    n_samples: int,
    radius: float,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample offsets for node/junction disks, balls, circles, or spheres."""

    if n_samples <= 0:
        return np.empty((0, int(dimension)), dtype=float)
    if radius <= 0.0 or mode == "centerline":
        return np.zeros((n_samples, int(dimension)), dtype=float)
    return _sample_ball_or_sphere(
        n_samples=n_samples,
        dimension=int(dimension),
        radius=radius,
        mode=mode,
        rng=rng,
    )


def _orthonormal_complement(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    dimension = int(direction.shape[0])
    norm = float(np.linalg.norm(direction))
    if dimension < 2:
        raise ValueError("Tube sampling requires ambient dimension at least 2.")
    if norm <= np.finfo(float).eps:
        return np.eye(dimension, dtype=float)[:, 1:]

    unit = direction / norm
    matrix = np.column_stack([unit, np.eye(dimension, dtype=float)])
    q, _ = np.linalg.qr(matrix)
    return q[:, 1:]


def _sample_ball_or_sphere(
    n_samples: int,
    dimension: int,
    radius: float,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if dimension <= 0:
        return np.zeros((n_samples, 0), dtype=float)
    if mode == "solid":
        if dimension == 1:
            return rng.uniform(-radius, radius, size=(n_samples, 1))
        directions = _sample_unit_vectors(n_samples, dimension, rng)
        radial = radius * rng.random(n_samples) ** (1.0 / float(dimension))
        return directions * radial[:, None]
    if mode == "surface":
        if dimension == 1:
            signs = rng.choice(np.array([-1.0, 1.0]), size=n_samples)
            return (radius * signs)[:, None]
        return radius * _sample_unit_vectors(n_samples, dimension, rng)
    raise ValueError(f"Unsupported tube sampling mode {mode!r}.")


def _sample_unit_vectors(
    n_samples: int,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = rng.normal(size=(n_samples, dimension))
    norms = np.linalg.norm(values, axis=1)
    zero = norms <= np.finfo(float).eps
    while np.any(zero):
        values[zero] = rng.normal(size=(int(np.sum(zero)), dimension))
        norms = np.linalg.norm(values, axis=1)
        zero = norms <= np.finfo(float).eps
    return values / norms[:, None]


def add_gaussian_noise(
    points: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Add isotropic Gaussian noise to sampled points."""

    if points.size == 0:
        return points.copy(), np.empty_like(points)
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")
    if sigma == 0.0:
        noise = np.zeros_like(points)
        return points.copy(), noise

    noise = rng.normal(loc=0.0, scale=sigma, size=points.shape)
    return points + noise, noise


def validate_sampled_point_cloud(
    G: nx.Graph,
    clean_points: np.ndarray,
    noisy_points: np.ndarray,
    sigma: float,
    include_node_points: bool,
    num_edge_sampled_points: int,
) -> None:
    """Run lightweight validation checks on the sampled point cloud."""

    if clean_points.ndim != 2:
        raise ValueError(f"clean_points must have shape (N, D), got {clean_points.shape}.")
    if noisy_points.ndim != 2:
        raise ValueError(f"noisy_points must have shape (N, D), got {noisy_points.shape}.")
    if clean_points.shape[1] < 2:
        raise ValueError("clean_points must have ambient dimension at least 2.")
    if noisy_points.shape[1] != clean_points.shape[1]:
        raise ValueError("clean_points and noisy_points must live in the same dimension.")
    if clean_points.shape != noisy_points.shape:
        raise ValueError("clean_points and noisy_points must have the same shape.")
    if not np.all(np.isfinite(clean_points)):
        raise ValueError("clean_points contains NaN or inf values.")
    if not np.all(np.isfinite(noisy_points)):
        raise ValueError("noisy_points contains NaN or inf values.")
    if sigma == 0.0 and not np.allclose(clean_points, noisy_points):
        raise ValueError("sigma is zero but noisy_points differs from clean_points.")
    if include_node_points:
        minimum_expected = G.number_of_nodes() + num_edge_sampled_points
        if clean_points.shape[0] < minimum_expected:
            raise ValueError("Point cloud is missing expected node or edge samples.")
