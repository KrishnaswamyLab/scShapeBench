"""Constructive embedding for the graph-generation pipeline."""

from __future__ import annotations

from collections import deque
import math
import random
from typing import Any

import networkx as nx
import numpy as np

from .geometry2d import (
    PlacementError,
    angularly_spread_child_directions,
    candidate_branch_clearance,
    choose_outward_direction,
    place_cycle_polygon,
    place_cycle_polygon_from_anchor,
    place_polyline_between_anchors,
    place_polyline_path,
    point_segment_distance_2d,
    polygon_radius_for_cycle,
    segment_distance_2d,
    validate_embedding,
    validate_partial_embedding,
)
from .geometry3d import (
    candidate_branch_clearance_3d,
    choose_outward_direction_3d,
    place_cycle_polygon_3d,
    place_cycle_polygon_from_anchor_3d,
    place_polyline_between_anchors_3d,
    place_polyline_path_3d,
    random_unit_vector_3d,
    spread_child_directions_3d,
    validate_embedding_3d,
    validate_partial_embedding_3d,
)
from .geometry_nd import (
    candidate_branch_clearance_nd,
    choose_outward_direction_nd,
    inflate_complement_positions_nd,
    normalize_nd,
    place_cycle_polygon_from_anchor_nd,
    place_cycle_polygon_nd,
    place_polyline_between_anchors_nd,
    place_polyline_path_nd,
    random_orthogonal_unit,
    random_unit_vector_nd,
    sample_branch_direction_nd,
    spread_child_directions_nd,
    validate_embedding_nd,
    validate_partial_embedding_nd,
)
from .graph_primitives import _edge_key
from .metadata_utils import (
    build_node_cycle_membership,
    compute_backbone_nodes,
    cycle_edge_lengths,
    extract_attachment_map_from_metadata,
    extract_branch_children_from_metadata,
    extract_cycle_ordering_from_metadata,
)

__all__ = [
    "apply_pca_crowding_to_embedding",
    "apply_pca_crowding_by_relative_budget",
    "estimate_min_admissible_crowding_gamma",
    "apply_pca_crowding_to_target_separation_ratio",
    "assign_target_edge_lengths",
    "compute_initial_cycle_backbone_geometry",
    "embed_graph_constructively",
    "embed_graph_constructively_2d",
    "fallback_planar_embedding",
    "fallback_spring_embedding",
    "place_branch_subtrees_constructively",
    "refine_embedding_with_anchored_stress",
    "sample_embedding_params",
]


def embed_graph_constructively(
    G: nx.Graph,
    metadata: dict[str, Any] | None = None,
    feature_scale_range: tuple[float, float] = (1.0, 1.0),
    separation_to_feature_ratio_range: tuple[float, float] = (0.8, 2.5),
    packing_to_feature_ratio_range: tuple[float, float] | None = None,
    noise_to_feature_ratio_range: tuple[float, float] = (0.02, 0.25),
    sampling_density_per_feature_range: tuple[float, float] = (10.0, 40.0),
    edge_length_jitter_ratio_range: tuple[float, float] = (0.0, 0.15),
    branch_angle_spread_range: tuple[float, float] = (0.5, 1.4),
    max_local_placement_attempts: int = 40,
    max_global_resampling_attempts: int = 20,
    embedding_dimensions: tuple[int, ...] | None = None,
    max_embedding_dimension: int = 5,
    hard_min_clearance_to_feature_ratio: float = 0.2,
    refinement_enabled: bool = True,
    refinement_max_iterations: int = 24,
    seed: int | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Constructively embed a cycle-and-branch graph in a bounded dimension ladder."""

    dimensions = _resolve_embedding_dimensions(
        embedding_dimensions=embedding_dimensions,
        max_embedding_dimension=max_embedding_dimension,
    )

    if G.number_of_nodes() == 0:
        output_metadata = dict(metadata) if metadata is not None else {}
        output_metadata["embedding_params"] = {}
        output_metadata["node_positions"] = {}
        output_metadata["node_positions_2d"] = {}
        output_metadata["embedding_valid"] = True
        output_metadata["embedding_dimension"] = None
        output_metadata["embedding_dimensions_attempted"] = list(dimensions)
        output_metadata["embedding_attempt_history"] = []
        output_metadata["embedding_constructive_success"] = True
        output_metadata["embedding_rejected"] = False
        output_metadata["min_nonadjacent_node_distance"] = math.inf
        output_metadata["min_nonincident_edge_distance"] = math.inf
        output_metadata["min_nonincident_node_edge_distance"] = math.inf
        output_metadata["global_embedding_attempts"] = 0
        output_metadata["total_global_embedding_attempts"] = 0
        output_metadata["local_placement_rejections"] = 0
        output_metadata["embedding_fallback_used"] = False
        output_metadata["embedding_fallback_mode"] = None
        return {}, output_metadata

    base_rng = random.Random(seed)
    param_rng = random.Random(base_rng.randrange(1 << 63))
    length_rng = random.Random(base_rng.randrange(1 << 63))
    placement_rngs = {dimension: random.Random(base_rng.randrange(1 << 63)) for dimension in dimensions}
    base_metadata = dict(metadata) if metadata is not None else {}

    embedding_params = sample_embedding_params(
        feature_scale_range=feature_scale_range,
        separation_to_feature_ratio_range=separation_to_feature_ratio_range,
        packing_to_feature_ratio_range=packing_to_feature_ratio_range,
        noise_to_feature_ratio_range=noise_to_feature_ratio_range,
        sampling_density_per_feature_range=sampling_density_per_feature_range,
        edge_length_jitter_ratio_range=edge_length_jitter_ratio_range,
        branch_angle_spread_range=branch_angle_spread_range,
        hard_min_clearance_to_feature_ratio=hard_min_clearance_to_feature_ratio,
        rng=param_rng,
    )
    edge_lengths = assign_target_edge_lengths(
        G=G,
        feature_scale=embedding_params["feature_scale"],
        jitter_ratio=embedding_params["edge_length_jitter_ratio"],
        rng=length_rng,
    )
    context = _prepare_embedding_context(G, base_metadata)

    attempt_history: list[dict[str, Any]] = []
    total_local_rejections = 0
    last_attempt_meta: dict[str, Any] | None = None
    last_attempt_dimension: int | None = None

    for dimension in dimensions:
        pos, attempt_meta = _run_constructive_embedding_nd_with_params(
            G=G,
            context=context,
            edge_lengths=edge_lengths,
            embedding_params=embedding_params,
            dimension=dimension,
            max_local_placement_attempts=max_local_placement_attempts,
            max_global_resampling_attempts=max_global_resampling_attempts,
            refinement_enabled=refinement_enabled,
            refinement_max_iterations=refinement_max_iterations,
            rng=placement_rngs[dimension],
        )

        total_local_rejections += attempt_meta["local_placement_rejections"]
        attempt_history.append(
            {
                "dimension": dimension,
                "embedding_valid": attempt_meta["embedding_valid"],
                "global_embedding_attempts": attempt_meta["global_embedding_attempts"],
                "local_placement_rejections": attempt_meta["local_placement_rejections"],
                "min_nonadjacent_node_distance": attempt_meta["min_nonadjacent_node_distance"],
                "min_nonincident_edge_distance": attempt_meta["min_nonincident_edge_distance"],
                "min_nonincident_node_edge_distance": attempt_meta[
                    "min_nonincident_node_edge_distance"
                ],
                "failure_reason": attempt_meta.get("failure_reason"),
                "refinement_applied": attempt_meta["embedding_refinement"]["applied"],
                "refinement_iterations": attempt_meta["embedding_refinement"]["iterations"],
                "refinement_objective_delta": attempt_meta["embedding_refinement"][
                    "objective_delta"
                ],
                "refinement_clearance_delta": attempt_meta["embedding_refinement"][
                    "clearance_delta"
                ],
            }
        )
        last_attempt_meta = attempt_meta
        last_attempt_dimension = dimension

        if attempt_meta["embedding_valid"]:
            output_metadata = dict(base_metadata)
            output_metadata["embedding_params"] = embedding_params
            output_metadata["node_positions"] = _serialize_node_positions(pos)
            output_metadata[f"node_positions_{dimension}d"] = _serialize_node_positions(pos)
            if dimension == 2:
                output_metadata["node_positions_2d"] = _serialize_node_positions(pos)
            elif dimension == 3:
                output_metadata["node_positions_3d"] = _serialize_node_positions(pos)
            output_metadata["embedding_valid"] = True
            output_metadata["embedding_dimension"] = dimension
            output_metadata["embedding_dimensions_attempted"] = [item["dimension"] for item in attempt_history]
            output_metadata["embedding_attempt_history"] = attempt_history
            output_metadata["embedding_constructive_success"] = True
            output_metadata["embedding_rejected"] = False
            output_metadata["min_nonadjacent_node_distance"] = attempt_meta[
                "min_nonadjacent_node_distance"
            ]
            output_metadata["min_nonincident_edge_distance"] = attempt_meta[
                "min_nonincident_edge_distance"
            ]
            output_metadata["min_nonincident_node_edge_distance"] = attempt_meta[
                "min_nonincident_node_edge_distance"
            ]
            output_metadata["global_embedding_attempts"] = attempt_meta[
                "global_embedding_attempts"
            ]
            output_metadata["total_global_embedding_attempts"] = sum(
                item["global_embedding_attempts"] for item in attempt_history
            )
            output_metadata["local_placement_rejections"] = total_local_rejections
            output_metadata["embedding_fallback_used"] = False
            output_metadata["embedding_fallback_mode"] = None
            output_metadata["embedding_refinement"] = attempt_meta["embedding_refinement"]
            return pos, output_metadata

    output_metadata = dict(base_metadata)
    output_metadata["embedding_params"] = embedding_params
    output_metadata["node_positions"] = {}
    output_metadata["node_positions_2d"] = {}
    output_metadata["embedding_valid"] = False
    output_metadata["embedding_dimension"] = None
    output_metadata["embedding_dimensions_attempted"] = [item["dimension"] for item in attempt_history]
    output_metadata["embedding_attempt_history"] = attempt_history
    output_metadata["embedding_constructive_success"] = False
    output_metadata["embedding_rejected"] = True
    output_metadata["embedding_rejection_reason"] = (
        "constructive_embedding_failed_in_all_dimensions"
    )
    output_metadata["min_nonadjacent_node_distance"] = (
        last_attempt_meta["min_nonadjacent_node_distance"] if last_attempt_meta else 0.0
    )
    output_metadata["min_nonincident_edge_distance"] = (
        last_attempt_meta["min_nonincident_edge_distance"] if last_attempt_meta else 0.0
    )
    output_metadata["min_nonincident_node_edge_distance"] = (
        last_attempt_meta["min_nonincident_node_edge_distance"] if last_attempt_meta else 0.0
    )
    output_metadata["global_embedding_attempts"] = (
        last_attempt_meta["global_embedding_attempts"] if last_attempt_meta else 0
    )
    output_metadata["total_global_embedding_attempts"] = sum(
        item["global_embedding_attempts"] for item in attempt_history
    )
    output_metadata["local_placement_rejections"] = total_local_rejections
    output_metadata["embedding_fallback_used"] = False
    output_metadata["embedding_fallback_mode"] = None
    if last_attempt_dimension is not None:
        output_metadata["last_failed_embedding_dimension"] = last_attempt_dimension
    if last_attempt_meta and last_attempt_meta.get("failure_reason"):
        output_metadata["last_failed_embedding_reason"] = last_attempt_meta["failure_reason"]
    return {}, output_metadata


def embed_graph_constructively_2d(
    G: nx.Graph,
    metadata: dict[str, Any] | None = None,
    feature_scale_range: tuple[float, float] = (1.0, 1.0),
    separation_to_feature_ratio_range: tuple[float, float] = (0.8, 2.5),
    packing_to_feature_ratio_range: tuple[float, float] | None = None,
    noise_to_feature_ratio_range: tuple[float, float] = (0.02, 0.25),
    sampling_density_per_feature_range: tuple[float, float] = (10.0, 40.0),
    edge_length_jitter_ratio_range: tuple[float, float] = (0.0, 0.15),
    branch_angle_spread_range: tuple[float, float] = (0.5, 1.4),
    max_local_placement_attempts: int = 40,
    max_global_resampling_attempts: int = 20,
    fallback_mode: str | None = None,
    hard_min_clearance_to_feature_ratio: float = 0.2,
    seed: int | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Constructively embed a graph in 2D only, without fallback layouts."""

    del fallback_mode
    return embed_graph_constructively(
        G=G,
        metadata=metadata,
        feature_scale_range=feature_scale_range,
        separation_to_feature_ratio_range=separation_to_feature_ratio_range,
        packing_to_feature_ratio_range=packing_to_feature_ratio_range,
        noise_to_feature_ratio_range=noise_to_feature_ratio_range,
        sampling_density_per_feature_range=sampling_density_per_feature_range,
        edge_length_jitter_ratio_range=edge_length_jitter_ratio_range,
        branch_angle_spread_range=branch_angle_spread_range,
        max_local_placement_attempts=max_local_placement_attempts,
        max_global_resampling_attempts=max_global_resampling_attempts,
        embedding_dimensions=(2,),
        max_embedding_dimension=2,
        hard_min_clearance_to_feature_ratio=hard_min_clearance_to_feature_ratio,
        seed=seed,
    )


def apply_pca_crowding_to_embedding(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    metadata: dict[str, Any] | None = None,
    gamma: float = 1.0,
    isotropy_epsilon: float = 0.05,
    crowding_model: dict[str, Any] | None = None,
    seed: int | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Apply the PCA-based crowding operator to an accepted embedding."""

    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must lie in (0, 1], got {gamma}.")
    if isotropy_epsilon < 0.0:
        raise ValueError("isotropy_epsilon must be non-negative.")

    output_metadata = dict(metadata) if metadata is not None else {}
    if not pos:
        output_metadata["crowding_applied"] = False
        output_metadata["crowding_gamma"] = gamma
        return {}, output_metadata

    model = (
        crowding_model
        if crowding_model is not None
        else _build_pca_crowding_model(
            pos=pos,
            isotropy_epsilon=isotropy_epsilon,
            seed=seed,
        )
    )
    crowded_pos, operator = _apply_pca_crowding_model(
        pos=pos,
        crowding_model=model,
        gamma=gamma,
    )
    dimension = int(model["dimension"])
    principal_axis = np.asarray(model["principal_axis"], dtype=float)
    centroid = np.asarray(model["centroid"], dtype=float)
    eigenvalues = np.asarray(model["eigenvalues"], dtype=float)
    covariance = np.asarray(model["covariance"], dtype=float)
    isotropic_fallback = bool(model["isotropic_fallback"])

    pre_stats = _compute_embedding_stats(G=G, pos=pos, dimension=dimension)
    post_stats = _compute_embedding_stats(G=G, pos=crowded_pos, dimension=dimension)
    pre_edge_summary = _compute_edge_length_summary(G=G, pos=pos)
    post_edge_summary = _compute_edge_length_summary(G=G, pos=crowded_pos)
    nominal_feature_scale = _extract_feature_scale_from_metadata(output_metadata)
    feature_scale_pre = (
        pre_edge_summary["median"] if pre_edge_summary["median"] > 0.0 else nominal_feature_scale
    )
    realized_sep = _min_structural_separation(post_stats)
    realized_sep_to_feature = (
        realized_sep / feature_scale_pre if feature_scale_pre > 0.0 and math.isfinite(realized_sep) else math.inf
    )
    realized_sep_to_post_edge = (
        realized_sep / post_edge_summary["median"]
        if post_edge_summary["median"] > 0.0 and math.isfinite(realized_sep)
        else math.inf
    )

    output_metadata["crowding_applied"] = True
    output_metadata["crowding_gamma"] = float(gamma)
    output_metadata["crowding_isotropy_epsilon"] = float(isotropy_epsilon)
    output_metadata["crowding_axis"] = principal_axis.tolist()
    output_metadata["crowding_centroid"] = centroid.tolist()
    output_metadata["crowding_covariance_eigenvalues"] = eigenvalues.tolist()
    output_metadata["crowding_covariance_matrix"] = covariance.tolist()
    output_metadata["crowding_isotropic_fallback_used"] = isotropic_fallback
    output_metadata["crowding_linear_operator"] = operator.tolist()
    output_metadata["crowding_model_dimension"] = dimension
    output_metadata["feature_scale_nominal"] = nominal_feature_scale
    output_metadata["feature_scale_pre_crowding"] = feature_scale_pre
    output_metadata["feature_scale_post_crowding_median_edge"] = post_edge_summary["median"]
    output_metadata["pre_crowding_edge_length_summary"] = pre_edge_summary
    output_metadata["post_crowding_edge_length_summary"] = post_edge_summary
    output_metadata["pre_crowding_min_nonadjacent_node_distance"] = pre_stats["min_nonadjacent_node_distance"]
    output_metadata["pre_crowding_min_nonincident_edge_distance"] = pre_stats["min_nonincident_edge_distance"]
    output_metadata["pre_crowding_min_nonincident_node_edge_distance"] = pre_stats["min_nonincident_node_edge_distance"]
    output_metadata["realized_post_crowding_min_nonadjacent_node_distance"] = post_stats["min_nonadjacent_node_distance"]
    output_metadata["realized_post_crowding_min_nonincident_edge_distance"] = post_stats["min_nonincident_edge_distance"]
    output_metadata["realized_post_crowding_min_nonincident_node_edge_distance"] = post_stats["min_nonincident_node_edge_distance"]
    output_metadata["realized_post_crowding_min_structural_distance"] = realized_sep
    output_metadata["realized_post_crowding_separation_to_feature_scale"] = realized_sep_to_feature
    output_metadata["realized_post_crowding_separation_to_post_median_edge"] = realized_sep_to_post_edge
    output_metadata["realized_post_crowding_absolute_hardness"] = (
        feature_scale_pre / realized_sep if realized_sep > 0.0 and math.isfinite(realized_sep) else math.inf
    )
    output_metadata["node_positions_pre_crowding"] = _serialize_node_positions(pos)
    output_metadata["node_positions"] = _serialize_node_positions(crowded_pos)
    if dimension == 2:
        output_metadata["node_positions_2d"] = _serialize_node_positions(crowded_pos)
        output_metadata["node_positions_2d_pre_crowding"] = _serialize_node_positions(pos)
    elif dimension == 3:
        output_metadata["node_positions_3d"] = _serialize_node_positions(crowded_pos)
        output_metadata["node_positions_3d_pre_crowding"] = _serialize_node_positions(pos)
    else:
        output_metadata[f"node_positions_{dimension}d"] = _serialize_node_positions(crowded_pos)
        output_metadata[f"node_positions_{dimension}d_pre_crowding"] = _serialize_node_positions(pos)
    output_metadata["embedding_dimension"] = dimension
    return crowded_pos, output_metadata


def apply_pca_crowding_to_target_separation_ratio(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    metadata: dict[str, Any] | None = None,
    target_separation_ratio: float = 0.5,
    isotropy_epsilon: float = 0.05,
    min_gamma: float = 0.05,
    max_gamma: float = 1.0,
    search_tolerance: float = 0.01,
    max_search_iters: int = 18,
    seed: int | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Find a gamma that best matches a target realized post-crowding separation ratio."""

    if target_separation_ratio <= 0.0:
        raise ValueError("target_separation_ratio must be positive.")
    if not (0.0 < min_gamma <= max_gamma <= 1.0):
        raise ValueError("Require 0 < min_gamma <= max_gamma <= 1.")
    if search_tolerance < 0.0:
        raise ValueError("search_tolerance must be non-negative.")
    if max_search_iters < 1:
        raise ValueError("max_search_iters must be at least 1.")

    model = _build_pca_crowding_model(
        pos=pos,
        isotropy_epsilon=isotropy_epsilon,
        seed=seed,
    )
    low_pos, low_meta = apply_pca_crowding_to_embedding(
        G=G,
        pos=pos,
        metadata=metadata,
        gamma=min_gamma,
        isotropy_epsilon=isotropy_epsilon,
        crowding_model=model,
        seed=seed,
    )
    high_pos, high_meta = apply_pca_crowding_to_embedding(
        G=G,
        pos=pos,
        metadata=metadata,
        gamma=max_gamma,
        isotropy_epsilon=isotropy_epsilon,
        crowding_model=model,
        seed=seed,
    )
    low_ratio = float(low_meta["realized_post_crowding_separation_to_feature_scale"])
    high_ratio = float(high_meta["realized_post_crowding_separation_to_feature_scale"])
    search_history = [
        {"gamma": min_gamma, "realized_separation_ratio": low_ratio},
        {"gamma": max_gamma, "realized_separation_ratio": high_ratio},
    ]

    if target_separation_ratio <= low_ratio:
        chosen_pos, chosen_meta = low_pos, low_meta
    elif target_separation_ratio >= high_ratio:
        chosen_pos, chosen_meta = high_pos, high_meta
    else:
        low = min_gamma
        high = max_gamma
        chosen_pos, chosen_meta = high_pos, high_meta
        best_error = abs(high_ratio - target_separation_ratio)
        for _ in range(max_search_iters):
            mid = 0.5 * (low + high)
            mid_pos, mid_meta = apply_pca_crowding_to_embedding(
                G=G,
                pos=pos,
                metadata=metadata,
                gamma=mid,
                isotropy_epsilon=isotropy_epsilon,
                crowding_model=model,
                seed=seed,
            )
            mid_ratio = float(mid_meta["realized_post_crowding_separation_to_feature_scale"])
            search_history.append(
                {"gamma": mid, "realized_separation_ratio": mid_ratio}
            )
            mid_error = abs(mid_ratio - target_separation_ratio)
            if mid_error <= best_error:
                best_error = mid_error
                chosen_pos, chosen_meta = mid_pos, mid_meta
            if abs(mid_ratio - target_separation_ratio) <= search_tolerance:
                break
            if mid_ratio < target_separation_ratio:
                low = mid
            else:
                high = mid

    chosen_meta = dict(chosen_meta)
    chosen_meta["crowding_target_separation_ratio"] = float(target_separation_ratio)
    chosen_meta["crowding_gamma_search_range"] = [float(min_gamma), float(max_gamma)]
    chosen_meta["crowding_gamma_search_tolerance"] = float(search_tolerance)
    chosen_meta["crowding_gamma_search_iterations"] = len(search_history)
    chosen_meta["crowding_gamma_search_history"] = search_history
    chosen_meta["crowding_target_separation_ratio_error"] = (
        float(chosen_meta["realized_post_crowding_separation_to_feature_scale"]) - float(target_separation_ratio)
    )
    return chosen_pos, chosen_meta


def estimate_min_admissible_crowding_gamma(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    metadata: dict[str, Any] | None = None,
    isotropy_epsilon: float = 0.05,
    epsilon_sep: float = 0.02,
    min_gamma: float = 0.01,
    search_tolerance: float = 0.005,
    max_search_iters: int = 24,
    seed: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Estimate the smallest admissible gamma before structural near-contact."""

    if epsilon_sep <= 0.0:
        raise ValueError("epsilon_sep must be positive.")
    if not (0.0 < min_gamma <= 1.0):
        raise ValueError("Require 0 < min_gamma <= 1.")
    if search_tolerance < 0.0:
        raise ValueError("search_tolerance must be non-negative.")
    if max_search_iters < 1:
        raise ValueError("max_search_iters must be at least 1.")

    model = _build_pca_crowding_model(
        pos=pos,
        isotropy_epsilon=isotropy_epsilon,
        seed=seed,
    )
    base_meta = dict(metadata) if metadata is not None else {}

    high_pos, high_meta = apply_pca_crowding_to_embedding(
        G=G,
        pos=pos,
        metadata=base_meta,
        gamma=1.0,
        isotropy_epsilon=isotropy_epsilon,
        crowding_model=model,
        seed=seed,
    )
    del high_pos
    high_admissible = _is_crowding_embedding_admissible(high_meta, epsilon_sep=epsilon_sep)
    if not high_admissible:
        raise ValueError("Clean embedding is not admissible under the requested crowding epsilon.")

    low_pos, low_meta = apply_pca_crowding_to_embedding(
        G=G,
        pos=pos,
        metadata=base_meta,
        gamma=min_gamma,
        isotropy_epsilon=isotropy_epsilon,
        crowding_model=model,
        seed=seed,
    )
    del low_pos
    low_ratio = float(low_meta["realized_post_crowding_separation_to_feature_scale"])
    high_ratio = float(high_meta["realized_post_crowding_separation_to_feature_scale"])
    search_history = [
        {
            "gamma": float(min_gamma),
            "admissible": _is_crowding_embedding_admissible(low_meta, epsilon_sep=epsilon_sep),
            "realized_separation_ratio": low_ratio,
        },
        {
            "gamma": 1.0,
            "admissible": True,
            "realized_separation_ratio": high_ratio,
        },
    ]

    low_admissible = search_history[0]["admissible"]
    if low_admissible:
        result_metadata = dict(low_meta)
        result_metadata["crowding_gamma_min"] = float(min_gamma)
        result_metadata["crowding_admissibility_epsilon_sep"] = float(epsilon_sep)
        result_metadata["crowding_gamma_min_search_tolerance"] = float(search_tolerance)
        result_metadata["crowding_gamma_min_search_history"] = search_history
        result_metadata["crowding_gamma_min_search_iterations"] = len(search_history)
        result_metadata["crowding_gamma_min_search_bracket"] = [float(min_gamma), 1.0]
        return float(min_gamma), result_metadata

    low = min_gamma
    high = 1.0
    best_gamma = 1.0
    best_meta = dict(high_meta)
    for _ in range(max_search_iters):
        mid = 0.5 * (low + high)
        mid_pos, mid_meta = apply_pca_crowding_to_embedding(
            G=G,
            pos=pos,
            metadata=base_meta,
            gamma=mid,
            isotropy_epsilon=isotropy_epsilon,
            crowding_model=model,
            seed=seed,
        )
        del mid_pos
        admissible = _is_crowding_embedding_admissible(mid_meta, epsilon_sep=epsilon_sep)
        search_history.append(
            {
                "gamma": float(mid),
                "admissible": bool(admissible),
                "realized_separation_ratio": float(
                    mid_meta["realized_post_crowding_separation_to_feature_scale"]
                ),
            }
        )
        if admissible:
            best_gamma = mid
            best_meta = dict(mid_meta)
            high = mid
        else:
            low = mid
        if high - low <= search_tolerance:
            break

    best_meta["crowding_gamma_min"] = float(best_gamma)
    best_meta["crowding_admissibility_epsilon_sep"] = float(epsilon_sep)
    best_meta["crowding_gamma_min_search_tolerance"] = float(search_tolerance)
    best_meta["crowding_gamma_min_search_history"] = search_history
    best_meta["crowding_gamma_min_search_iterations"] = len(search_history)
    best_meta["crowding_gamma_min_search_bracket"] = [float(low), float(high)]
    return float(best_gamma), best_meta


def apply_pca_crowding_by_relative_budget(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    metadata: dict[str, Any] | None = None,
    r: float = 0.0,
    isotropy_epsilon: float = 0.05,
    epsilon_sep: float = 0.02,
    min_gamma: float = 0.01,
    search_tolerance: float = 0.005,
    max_search_iters: int = 24,
    gamma_min: float | None = None,
    gamma_min_metadata: dict[str, Any] | None = None,
    seed: int | None = None,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Apply PCA crowding using normalized relative crowding budget r in [0, 1]."""

    if not (0.0 <= r <= 1.0):
        raise ValueError(f"r must lie in [0, 1], got {r}.")

    if gamma_min is None:
        gamma_min, gamma_min_meta = estimate_min_admissible_crowding_gamma(
            G=G,
            pos=pos,
            metadata=metadata,
            isotropy_epsilon=isotropy_epsilon,
            epsilon_sep=epsilon_sep,
            min_gamma=min_gamma,
            search_tolerance=search_tolerance,
            max_search_iters=max_search_iters,
            seed=seed,
        )
    else:
        gamma_min_meta = dict(gamma_min_metadata) if gamma_min_metadata is not None else dict(metadata or {})
        gamma_min_meta["crowding_gamma_min"] = float(gamma_min)
        gamma_min_meta["crowding_admissibility_epsilon_sep"] = float(epsilon_sep)

    gamma = 1.0 - float(r) * (1.0 - float(gamma_min))
    crowded_pos, crowded_meta = apply_pca_crowding_to_embedding(
        G=G,
        pos=pos,
        metadata=gamma_min_meta,
        gamma=gamma,
        isotropy_epsilon=isotropy_epsilon,
        seed=seed,
    )
    crowded_meta = dict(crowded_meta)
    crowded_meta["crowding_r"] = float(r)
    crowded_meta["crowding_gamma_min"] = float(gamma_min)
    crowded_meta["crowding_admissibility_epsilon_sep"] = float(epsilon_sep)
    crowded_meta["crowding_relative_budget_definition"] = "gamma = 1 - r * (1 - gamma_min)"
    return crowded_pos, crowded_meta


def _resolve_embedding_dimensions(
    *,
    embedding_dimensions: tuple[int, ...] | None,
    max_embedding_dimension: int,
) -> list[int]:
    if max_embedding_dimension < 2 or max_embedding_dimension > 5:
        raise ValueError("max_embedding_dimension must lie in [2, 5].")

    if embedding_dimensions is None:
        return list(range(2, max_embedding_dimension + 1))

    dimensions: list[int] = []
    for dimension in embedding_dimensions:
        if dimension < 2 or dimension > 5:
            raise ValueError(f"Unsupported embedding dimension {dimension}.")
        if dimension > max_embedding_dimension:
            raise ValueError(
                f"embedding dimension {dimension} exceeds max_embedding_dimension={max_embedding_dimension}."
            )
        if dimension not in dimensions:
            dimensions.append(dimension)
    if not dimensions:
        raise ValueError("embedding_dimensions must contain at least one supported dimension.")
    return dimensions


def _prepare_embedding_context(
    G: nx.Graph,
    base_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cycle_nodes": extract_cycle_ordering_from_metadata(G, base_metadata),
        "attachments_by_pair": extract_attachment_map_from_metadata(base_metadata),
        "cycle_tree_edges": [tuple(edge) for edge in base_metadata.get("cycle_tree_edges", [])],
        "branch_children": extract_branch_children_from_metadata(base_metadata),
        "branch_parent_map": dict(base_metadata.get("branch_parent_map", {})),
        "branch_depth_map": dict(base_metadata.get("branch_depth_map", {})),
        "initial_branch_roots": list(base_metadata.get("branch_attachment_nodes_initial", [])),
        "backbone_nodes": compute_backbone_nodes(G, base_metadata),
    }


def _serialize_node_positions(pos: dict[int, np.ndarray]) -> dict[int, list[float]]:
    return {node: coords.tolist() for node, coords in sorted(pos.items())}


def _run_constructive_embedding_nd_with_params(
    G: nx.Graph,
    context: dict[str, Any],
    edge_lengths: dict[tuple[int, int], float],
    embedding_params: dict[str, float],
    dimension: int,
    max_local_placement_attempts: int,
    max_global_resampling_attempts: int,
    refinement_enabled: bool,
    refinement_max_iterations: int,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    last_pos: dict[int, np.ndarray] = {}
    last_stats: dict[str, float] = {
        "min_nonadjacent_node_distance": 0.0,
        "min_nonincident_edge_distance": 0.0,
        "min_nonincident_node_edge_distance": 0.0,
    }
    total_local_rejections = 0
    last_error = "constructive_embedding_failed"

    for global_attempt in range(1, max_global_resampling_attempts + 1):
        local_rejections = 0
        try:
            pos, cycle_centers, local_rejections = compute_initial_cycle_backbone_geometry_nd(
                G=G,
                cycle_nodes=context["cycle_nodes"],
                cycle_tree_edges=context["cycle_tree_edges"],
                attachments_by_pair=context["attachments_by_pair"],
                edge_lengths=edge_lengths,
                feature_scale=embedding_params["feature_scale"],
                target_packing_distance=embedding_params["target_packing_distance"],
                hard_min_clearance=embedding_params["hard_min_clearance"],
                max_local_placement_attempts=max_local_placement_attempts,
                dimension=dimension,
                rng=rng,
            )

            if not pos:
                root = next(iter(G.nodes()))
                pos[root] = np.zeros(dimension, dtype=float)
                cycle_centers = {}

            local_rejections += place_branch_subtrees_constructively_nd(
                G=G,
                pos=pos,
                branch_children=context["branch_children"],
                branch_parent_map=context["branch_parent_map"],
                branch_depth_map=context["branch_depth_map"],
                initial_branch_roots=context["initial_branch_roots"],
                cycle_nodes=context["cycle_nodes"],
                cycle_centers=cycle_centers,
                backbone_nodes=context["backbone_nodes"],
                edge_lengths=edge_lengths,
                feature_scale=embedding_params["feature_scale"],
                target_packing_distance=embedding_params["target_packing_distance"],
                hard_min_clearance=embedding_params["hard_min_clearance"],
                branch_angle_spread=embedding_params["branch_angle_spread"],
                max_local_placement_attempts=max_local_placement_attempts,
                dimension=dimension,
                rng=rng,
            )

            if len(pos) != G.number_of_nodes():
                raise PlacementError("Embedding attempt did not place all graph nodes.")

            embedding_valid, stats = validate_embedding_nd(
                G=G,
                pos=pos,
                target_min_separation=embedding_params["hard_min_clearance"],
                tolerance=0.05 * embedding_params["feature_scale"],
            )
            refinement_meta = {
                "enabled": refinement_enabled,
                "applied": False,
                "iterations": 0,
                "objective_before": None,
                "objective_after": None,
                "objective_delta": 0.0,
                "clearance_before": _min_structural_separation(stats),
                "clearance_after": _min_structural_separation(stats),
                "clearance_delta": 0.0,
                "accepted_steps": 0,
                "reason": "refinement_disabled" if not refinement_enabled else "not_attempted",
            }
            last_pos = {node: coords.copy() for node, coords in pos.items()}
            last_stats = stats
            total_local_rejections += local_rejections
            if embedding_valid:
                if refinement_enabled:
                    pos, refinement_meta = refine_embedding_with_anchored_stress(
                        G=G,
                        pos=pos,
                        context=context,
                        edge_lengths=edge_lengths,
                        embedding_params=embedding_params,
                        dimension=dimension,
                        max_iterations=refinement_max_iterations,
                    )
                    embedding_valid, stats = validate_embedding_nd(
                        G=G,
                        pos=pos,
                        target_min_separation=embedding_params["hard_min_clearance"],
                        tolerance=0.05 * embedding_params["feature_scale"],
                    )
                    last_pos = {node: coords.copy() for node, coords in pos.items()}
                    last_stats = stats
                return pos, {
                    "embedding_valid": True,
                    **stats,
                    "global_embedding_attempts": global_attempt,
                    "local_placement_rejections": total_local_rejections,
                    "failure_reason": None,
                    "embedding_refinement": refinement_meta,
                }
            last_error = "global_validation_failed"
        except PlacementError as exc:
            total_local_rejections += local_rejections
            last_error = str(exc)
            continue

    return last_pos, {
        "embedding_valid": False,
        **last_stats,
        "global_embedding_attempts": max_global_resampling_attempts,
        "local_placement_rejections": total_local_rejections,
        "failure_reason": last_error,
        "embedding_refinement": {
            "enabled": refinement_enabled,
            "applied": False,
            "iterations": 0,
            "objective_before": None,
            "objective_after": None,
            "objective_delta": 0.0,
            "clearance_before": _min_structural_separation(last_stats),
            "clearance_after": _min_structural_separation(last_stats),
            "clearance_delta": 0.0,
            "accepted_steps": 0,
            "reason": "constructive_embedding_failed",
        },
    }


def refine_embedding_with_anchored_stress(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    context: dict[str, Any],
    edge_lengths: dict[tuple[int, int], float],
    embedding_params: dict[str, float],
    dimension: int,
    max_iterations: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Apply a small anchored stress refinement without breaking hard validity checks."""

    base_pos = {node: np.asarray(coords, dtype=float).copy() for node, coords in pos.items()}
    if max_iterations <= 0 or len(base_pos) < 2:
        base_stats = _compute_embedding_stats(G=G, pos=base_pos, dimension=dimension)
        return base_pos, {
            "enabled": True,
            "applied": False,
            "iterations": 0,
            "objective_before": None,
            "objective_after": None,
            "objective_delta": 0.0,
            "clearance_before": _min_structural_separation(base_stats),
            "clearance_after": _min_structural_separation(base_stats),
            "clearance_delta": 0.0,
            "accepted_steps": 0,
            "reason": "graph_too_small",
        }

    hard_min_clearance = float(embedding_params["hard_min_clearance"])
    feature_scale = float(embedding_params["feature_scale"])
    preferred_min_separation = float(embedding_params["preferred_min_separation"])
    valid, base_stats = validate_embedding_nd(
        G=G,
        pos=base_pos,
        target_min_separation=hard_min_clearance,
        tolerance=0.05 * feature_scale,
    )
    base_clearance = _min_structural_separation(base_stats)
    if not valid:
        return base_pos, {
            "enabled": True,
            "applied": False,
            "iterations": 0,
            "objective_before": None,
            "objective_after": None,
            "objective_delta": 0.0,
            "clearance_before": base_clearance,
            "clearance_after": base_clearance,
            "clearance_delta": 0.0,
            "accepted_steps": 0,
            "reason": "invalid_input_embedding",
        }

    nodes = sorted(base_pos)
    index_of = {node: idx for idx, node in enumerate(nodes)}
    x0 = np.stack([base_pos[node] for node in nodes], axis=0)
    current = x0.copy()
    branch_depth_map = context.get("branch_depth_map", {})
    backbone_nodes = set(context.get("backbone_nodes", set()))
    initial_branch_roots = set(context.get("initial_branch_roots", []))

    anchor_weights = np.zeros(len(nodes), dtype=float)
    max_drift = np.zeros(len(nodes), dtype=float)
    for idx, node in enumerate(nodes):
        if node in backbone_nodes:
            anchor_weights[idx] = 10.0
            max_drift[idx] = 0.18 * feature_scale
        elif node in initial_branch_roots:
            anchor_weights[idx] = 4.5
            max_drift[idx] = 0.34 * feature_scale
        elif branch_depth_map.get(node, 0) >= 2:
            anchor_weights[idx] = 1.6
            max_drift[idx] = 0.60 * feature_scale
        else:
            anchor_weights[idx] = 2.5
            max_drift[idx] = 0.45 * feature_scale

    weighted_graph = nx.Graph()
    weighted_graph.add_nodes_from(G.nodes())
    weighted_graph.add_edges_from(
        (u, v, {"weight": float(edge_lengths[_edge_key(u, v)])}) for u, v in G.edges()
    )

    pair_i: list[int] = []
    pair_j: list[int] = []
    pair_target: list[float] = []
    pair_weight: list[float] = []
    repulsion_pairs: list[tuple[int, int]] = []
    for source in nodes:
        distances = nx.single_source_dijkstra_path_length(weighted_graph, source, weight="weight")
        for target in nodes:
            if source >= target:
                continue
            target_distance = float(distances[target])
            base_weight = 1.0 / max(target_distance * target_distance, 1e-9)
            if G.has_edge(source, target):
                weight = 6.0 * base_weight
            else:
                weight = 0.35 * base_weight
                repulsion_pairs.append((index_of[source], index_of[target]))
            pair_i.append(index_of[source])
            pair_j.append(index_of[target])
            pair_target.append(target_distance)
            pair_weight.append(weight)

    pair_i_arr = np.asarray(pair_i, dtype=int)
    pair_j_arr = np.asarray(pair_j, dtype=int)
    pair_target_arr = np.asarray(pair_target, dtype=float)
    pair_weight_arr = np.asarray(pair_weight, dtype=float)
    repulsion_weight = 3.0 / max(preferred_min_separation * preferred_min_separation, 1e-9)

    def to_pos_dict(coords: np.ndarray) -> dict[int, np.ndarray]:
        return {node: coords[index_of[node]].copy() for node in nodes}

    def objective(coords: np.ndarray) -> float:
        deltas = coords[pair_i_arr] - coords[pair_j_arr]
        distances = np.linalg.norm(deltas, axis=1)
        stress = np.sum(pair_weight_arr * np.square(distances - pair_target_arr))
        anchor = np.sum(anchor_weights[:, None] * np.square(coords - x0))
        repulsion = 0.0
        if repulsion_pairs:
            for idx_a, idx_b in repulsion_pairs:
                distance = float(np.linalg.norm(coords[idx_a] - coords[idx_b]))
                if distance < preferred_min_separation:
                    repulsion += repulsion_weight * (preferred_min_separation - distance) ** 2
        return float(stress + anchor + repulsion)

    def gradient(coords: np.ndarray) -> np.ndarray:
        grad = 2.0 * anchor_weights[:, None] * (coords - x0)
        deltas = coords[pair_i_arr] - coords[pair_j_arr]
        distances = np.linalg.norm(deltas, axis=1)
        safe_distances = np.maximum(distances, 1e-9)
        coeffs = 2.0 * pair_weight_arr * (distances - pair_target_arr) / safe_distances
        pair_grads = coeffs[:, None] * deltas
        np.add.at(grad, pair_i_arr, pair_grads)
        np.add.at(grad, pair_j_arr, -pair_grads)

        if repulsion_pairs:
            for idx_a, idx_b in repulsion_pairs:
                delta = coords[idx_a] - coords[idx_b]
                distance = float(np.linalg.norm(delta))
                if distance < 1e-9 or distance >= preferred_min_separation:
                    continue
                coeff = -2.0 * repulsion_weight * (preferred_min_separation - distance) / distance
                rep_grad = coeff * delta
                grad[idx_a] += rep_grad
                grad[idx_b] -= rep_grad
        return grad

    base_obj = objective(x0)
    current_obj = base_obj
    best_obj = base_obj
    best_coords = current.copy()
    best_stats = dict(base_stats)
    best_clearance = base_clearance
    step_size = 0.01
    max_step = 0.06 * feature_scale
    accepted_steps = 0
    attempted_iterations = 0

    for _ in range(max_iterations):
        attempted_iterations += 1
        grad = gradient(current)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < 1e-8:
            break

        local_step = step_size
        accepted = False
        while local_step >= 1e-5:
            delta = -local_step * grad
            delta_norms = np.linalg.norm(delta, axis=1)
            if float(np.max(delta_norms)) > max_step:
                delta *= max_step / float(np.max(delta_norms))

            proposal = current + delta
            total_drift = proposal - x0
            total_drift_norms = np.linalg.norm(total_drift, axis=1)
            drift_scale = np.minimum(1.0, max_drift / np.maximum(total_drift_norms, 1e-9))
            proposal = x0 + total_drift * drift_scale[:, None]

            proposal_pos = to_pos_dict(proposal)
            valid, proposal_stats = validate_embedding_nd(
                G=G,
                pos=proposal_pos,
                target_min_separation=hard_min_clearance,
                tolerance=0.05 * feature_scale,
            )
            if not valid:
                local_step *= 0.5
                continue

            proposal_obj = objective(proposal)
            if proposal_obj >= current_obj - 1e-7:
                local_step *= 0.5
                continue

            current = proposal
            current_obj = proposal_obj
            accepted_steps += 1
            accepted = True

            proposal_clearance = _min_structural_separation(proposal_stats)
            if proposal_obj < best_obj - 1e-6:
                best_obj = proposal_obj
                best_coords = proposal.copy()
                best_stats = proposal_stats
                best_clearance = proposal_clearance
            step_size = min(local_step * 1.05, 0.03)
            break

        if not accepted:
            step_size *= 0.5
            if step_size < 1e-5:
                break

    final_pos = to_pos_dict(best_coords)
    final_valid, final_stats = validate_embedding_nd(
        G=G,
        pos=final_pos,
        target_min_separation=hard_min_clearance,
        tolerance=0.05 * feature_scale,
    )
    final_clearance = _min_structural_separation(final_stats)
    if (
        final_valid
        and accepted_steps > 0
        and final_clearance >= base_clearance - 1e-6
        and best_obj < base_obj - 1e-6
    ):
        return final_pos, {
            "enabled": True,
            "applied": True,
            "iterations": attempted_iterations,
            "objective_before": float(base_obj),
            "objective_after": float(best_obj),
            "objective_delta": float(best_obj - base_obj),
            "clearance_before": base_clearance,
            "clearance_after": final_clearance,
            "clearance_delta": float(final_clearance - base_clearance),
            "accepted_steps": accepted_steps,
            "reason": "accepted",
        }

    return base_pos, {
        "enabled": True,
        "applied": False,
        "iterations": attempted_iterations,
        "objective_before": float(base_obj),
        "objective_after": float(best_obj),
        "objective_delta": float(best_obj - base_obj),
        "clearance_before": base_clearance,
        "clearance_after": final_clearance,
        "clearance_delta": float(final_clearance - base_clearance),
        "accepted_steps": accepted_steps,
        "reason": "no_admissible_improvement",
    }


def compute_initial_cycle_backbone_geometry_nd(
    G: nx.Graph,
    cycle_nodes: dict[int, list[int]],
    cycle_tree_edges: list[tuple[int, int]],
    attachments_by_pair: dict[tuple[int, int], dict[str, Any]],
    edge_lengths: dict[tuple[int, int], float],
    feature_scale: float,
    target_packing_distance: float,
    hard_min_clearance: float,
    max_local_placement_attempts: int,
    dimension: int,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]:
    """Constructively place the cycle backbone inside ``R^dimension``."""

    if not cycle_nodes:
        return {}, {}, 0

    pos: dict[int, np.ndarray] = {}
    cycle_centers: dict[int, np.ndarray] = {}
    local_rejections = 0

    root_cycle = min(cycle_nodes)
    root_nodes = cycle_nodes[root_cycle]
    root_edge_lengths = cycle_edge_lengths(root_nodes, edge_lengths)
    root_u = random_unit_vector_nd(dimension, rng)
    root_v = random_orthogonal_unit(root_u, rng)
    root_positions = place_cycle_polygon_nd(
        cycle_nodes=root_nodes,
        edge_lengths=root_edge_lengths,
        center=np.zeros(dimension, dtype=float),
        u_axis=root_u,
        v_axis=root_v,
        rotation=rng.uniform(0.0, 2.0 * math.pi),
    )
    pos.update(root_positions)
    cycle_centers[root_cycle] = np.mean(np.stack([pos[node] for node in root_nodes]), axis=0)

    tree_adj = _tree_adjacency(len(cycle_nodes), cycle_tree_edges)
    seen = {root_cycle}
    traversal = deque([root_cycle])

    while traversal:
        parent_cycle = traversal.popleft()
        parent_center = cycle_centers[parent_cycle]
        for child_cycle in tree_adj.get(parent_cycle, []):
            if child_cycle in seen:
                continue
            seen.add(child_cycle)
            traversal.append(child_cycle)

            pair = tuple(sorted((parent_cycle, child_cycle)))
            attachment = attachments_by_pair.get(pair)
            if attachment is None:
                raise PlacementError(f"Missing attachment metadata for cycle pair {pair}.")

            child_nodes = cycle_nodes[child_cycle]
            placed = False
            for _ in range(max_local_placement_attempts):
                if attachment["attachment_type"] == "path":
                    candidate_pos, child_center = _construct_path_attached_cycle_nd(
                        cycle_nodes=child_nodes,
                        attachment=attachment,
                        existing_pos=pos,
                        parent_center=parent_center,
                        edge_lengths=edge_lengths,
                        target_packing_distance=target_packing_distance,
                        dimension=dimension,
                        rng=rng,
                    )
                elif attachment["attachment_type"] == "shared_vertex":
                    candidate_pos, child_center = _construct_shared_vertex_cycle_nd(
                        cycle_nodes=child_nodes,
                        attachment=attachment,
                        existing_pos=pos,
                        parent_center=parent_center,
                        edge_lengths=edge_lengths,
                        target_packing_distance=target_packing_distance,
                        dimension=dimension,
                        rng=rng,
                    )
                elif attachment["attachment_type"] == "shared_path":
                    candidate_pos, child_center = _construct_shared_path_cycle_nd(
                        cycle_nodes=child_nodes,
                        attachment=attachment,
                        existing_pos=pos,
                        parent_center=parent_center,
                        edge_lengths=edge_lengths,
                        feature_scale=feature_scale,
                        target_packing_distance=target_packing_distance,
                        dimension=dimension,
                        rng=rng,
                    )
                else:
                    raise PlacementError(
                        f"Unsupported attachment type {attachment['attachment_type']}."
                    )

                if not validate_partial_embedding_nd(
                    G=G,
                    existing_pos=pos,
                    candidate_pos=candidate_pos,
                    target_min_separation=hard_min_clearance,
                    tolerance_ratio=0.15,
                ):
                    local_rejections += 1
                    continue

                pos.update(candidate_pos)
                cycle_centers[child_cycle] = child_center
                placed = True
                break

            if not placed:
                raise PlacementError(f"Failed to place child cycle {child_cycle}.")

    return pos, cycle_centers, local_rejections


def _construct_path_attached_cycle_nd(
    cycle_nodes: list[int],
    attachment: dict[str, Any],
    existing_pos: dict[int, np.ndarray],
    parent_center: np.ndarray,
    edge_lengths: dict[tuple[int, int], float],
    target_packing_distance: float,
    dimension: int,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    path_nodes = list(attachment["path_nodes"])
    anchor_a = attachment["anchor_a"]
    anchor_b = attachment["anchor_b"]
    if anchor_a not in existing_pos:
        raise PlacementError("Path attachment anchor on parent cycle is not placed.")

    child_edge_lengths = cycle_edge_lengths(cycle_nodes, edge_lengths)
    child_radius = polygon_radius_for_cycle(child_edge_lengths)
    direction = existing_pos[anchor_a] - parent_center
    if float(np.linalg.norm(direction)) < 1e-10:
        direction = random_unit_vector_nd(dimension, rng)
    direction = sample_branch_direction_nd(direction, angle_budget=0.9, rng=rng)

    path_edge_lengths = [
        edge_lengths[_edge_key(path_nodes[idx], path_nodes[idx + 1])]
        for idx in range(len(path_nodes) - 1)
    ]
    total_path_length = sum(path_edge_lengths)
    child_center = existing_pos[anchor_a] + direction * (
        total_path_length
        + child_radius
        + _cycle_packing_offset(
            target_packing_distance=target_packing_distance,
            feature_radius=child_radius,
        )
    )
    anchor_b_pos = existing_pos[anchor_a] + direction * total_path_length

    path_positions = place_polyline_path_nd(
        path_nodes=path_nodes,
        edge_lengths=path_edge_lengths,
        start_position=existing_pos[anchor_a],
        direction=direction,
        rng=rng,
        angular_jitter=0.0,
    )
    path_positions[anchor_b] = anchor_b_pos

    cycle_positions = place_cycle_polygon_from_anchor_nd(
        cycle_nodes=cycle_nodes,
        edge_lengths=child_edge_lengths,
        anchor_node=anchor_b,
        anchor_position=anchor_b_pos,
        center=child_center,
        rng=rng,
        v_axis_hint=random_orthogonal_unit(direction, rng),
    )
    candidate_pos = {
        node: coords
        for node, coords in {**path_positions, **cycle_positions}.items()
        if node not in existing_pos
    }
    return candidate_pos, child_center


def _construct_shared_vertex_cycle_nd(
    cycle_nodes: list[int],
    attachment: dict[str, Any],
    existing_pos: dict[int, np.ndarray],
    parent_center: np.ndarray,
    edge_lengths: dict[tuple[int, int], float],
    target_packing_distance: float,
    dimension: int,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    shared_node = attachment["shared_node"]
    if shared_node not in existing_pos:
        raise PlacementError("Shared vertex is not yet placed.")

    child_edge_lengths = cycle_edge_lengths(cycle_nodes, edge_lengths)
    child_radius = polygon_radius_for_cycle(child_edge_lengths)
    direction = existing_pos[shared_node] - parent_center
    if float(np.linalg.norm(direction)) < 1e-10:
        direction = random_unit_vector_nd(dimension, rng)
    direction = sample_branch_direction_nd(direction, angle_budget=1.0, rng=rng)
    child_center = existing_pos[shared_node] + direction * (
        child_radius
        + _cycle_packing_offset(
            target_packing_distance=target_packing_distance,
            feature_radius=child_radius,
        )
    )

    cycle_positions = place_cycle_polygon_from_anchor_nd(
        cycle_nodes=cycle_nodes,
        edge_lengths=child_edge_lengths,
        anchor_node=shared_node,
        anchor_position=existing_pos[shared_node],
        center=child_center,
        rng=rng,
        v_axis_hint=random_orthogonal_unit(direction, rng),
    )
    candidate_pos = {
        node: coords for node, coords in cycle_positions.items() if node not in existing_pos
    }
    return candidate_pos, child_center


def _construct_shared_path_cycle_nd(
    cycle_nodes: list[int],
    attachment: dict[str, Any],
    existing_pos: dict[int, np.ndarray],
    parent_center: np.ndarray,
    edge_lengths: dict[tuple[int, int], float],
    feature_scale: float,
    target_packing_distance: float,
    dimension: int,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    shared_nodes = list(attachment["shared_path_nodes"])
    for node in shared_nodes:
        if node not in existing_pos:
            raise PlacementError("Shared-path anchor nodes are not fully placed.")

    shared_in_cycle = _locate_path_in_cycle(cycle_nodes, shared_nodes)
    complement_path = _complementary_cycle_path(cycle_nodes, shared_in_cycle)
    edge_seq = [
        edge_lengths[_edge_key(complement_path[idx], complement_path[idx + 1])]
        for idx in range(len(complement_path) - 1)
    ]

    start_pos = existing_pos[complement_path[0]]
    end_pos = existing_pos[complement_path[-1]]
    midpoint = 0.5 * (start_pos + end_pos)
    chord = end_pos - start_pos
    tangent = normalize_nd(
        chord if float(np.linalg.norm(chord)) > 1e-12 else random_unit_vector_nd(dimension, rng)
    )
    preferred_normal = midpoint - parent_center
    preferred_normal = preferred_normal - float(np.dot(preferred_normal, tangent)) * tangent
    if float(np.linalg.norm(preferred_normal)) < 1e-10:
        preferred_normal = random_orthogonal_unit(tangent, rng)

    complement_positions = place_polyline_between_anchors_nd(
        path_nodes=complement_path,
        edge_lengths=edge_seq,
        start_position=start_pos,
        end_position=end_pos,
        preferred_normal=preferred_normal,
        feature_scale=feature_scale,
        rng=rng,
    )
    inflate_complement_positions_nd(
        complement_positions=complement_positions,
        complement_path=complement_path,
        preferred_normal=preferred_normal,
        edge_seq=edge_seq,
        target_packing_distance=target_packing_distance,
    )

    merged = dict(existing_pos)
    merged.update(complement_positions)
    child_center = np.mean(np.stack([merged[node] for node in cycle_nodes]), axis=0)
    return complement_positions, child_center


def place_branch_subtrees_constructively_nd(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    branch_children: dict[int, list[int]],
    branch_parent_map: dict[int, int],
    branch_depth_map: dict[int, int],
    initial_branch_roots: list[int],
    cycle_nodes: dict[int, list[int]],
    cycle_centers: dict[int, np.ndarray],
    backbone_nodes: set[int],
    edge_lengths: dict[tuple[int, int], float],
    feature_scale: float,
    target_packing_distance: float,
    hard_min_clearance: float,
    branch_angle_spread: float,
    max_local_placement_attempts: int,
    dimension: int,
    rng: random.Random,
) -> int:
    """Recursively place branch trees attached to the backbone in ``R^dimension``."""

    del branch_parent_map
    if not branch_children:
        return 0

    node_cycle_membership = build_node_cycle_membership(cycle_nodes)
    backbone_positions = [pos[node] for node in backbone_nodes if node in pos]
    if backbone_positions:
        backbone_centroid = np.mean(np.stack(backbone_positions), axis=0)
    else:
        backbone_centroid = np.zeros(dimension, dtype=float)
    local_rejections = 0

    def place_subtree(parent: int, incoming_direction: np.ndarray | None) -> list[int]:
        nonlocal local_rejections
        children = list(branch_children.get(parent, []))
        if not children:
            return []

        ordered_children = _order_branch_children(children, parent, branch_depth_map)
        preferred_direction = choose_outward_direction_nd(
            node=parent,
            pos=pos,
            cycle_membership=node_cycle_membership,
            cycle_centers=cycle_centers,
            backbone_centroid=backbone_centroid,
            incoming_direction=incoming_direction,
            rng=rng,
        )
        child_directions = spread_child_directions_nd(
            preferred_direction=preferred_direction,
            num_children=len(ordered_children),
            spread=_effective_branch_spread(
                branch_angle_spread=branch_angle_spread,
                target_packing_distance=target_packing_distance,
                feature_scale=feature_scale,
            ),
            rng=rng,
            continuation_child=ordered_children[0] if ordered_children else None,
            parent=parent,
            branch_depth_map=branch_depth_map,
        )

        placed_nodes: list[int] = []
        min_span = 0.35 if dimension == 2 else 0.25
        max_span = 1.35 if dimension == 2 else 1.15
        for child, base_direction in zip(ordered_children, child_directions):
            edge_length = edge_lengths[_edge_key(parent, child)]
            chosen_direction = None
            chosen_position = None

            for attempt_idx in range(max_local_placement_attempts):
                angle_budget = _interpolate_attempt_span(
                    attempt_idx=attempt_idx,
                    max_attempts=max_local_placement_attempts,
                    min_span=min_span,
                    max_span=max_span,
                )
                direction = sample_branch_direction_nd(
                    base_direction=base_direction,
                    angle_budget=angle_budget,
                    rng=rng,
                )
                candidate_position = pos[parent] + normalize_nd(direction) * edge_length
                candidate_pos = {child: candidate_position}

                if validate_partial_embedding_nd(
                    G=G,
                    existing_pos=pos,
                    candidate_pos=candidate_pos,
                    target_min_separation=hard_min_clearance,
                    tolerance_ratio=0.15,
                ):
                    chosen_direction = direction
                    chosen_position = candidate_position
                    break
                local_rejections += 1

            if chosen_direction is None or chosen_position is None:
                best_position = _best_branch_direction_nd(
                    G=G,
                    pos=pos,
                    parent=parent,
                    child=child,
                    base_direction=base_direction,
                    edge_length=edge_length,
                    max_local_placement_attempts=max_local_placement_attempts,
                    dimension=dimension,
                    rng=rng,
                )
                raise PlacementError(
                    f"Failed to place branch child {child} from parent {parent}; "
                    f"best clearance={candidate_branch_clearance_nd(G, pos, parent, child, best_position):.3f}."
                )

            pos[child] = chosen_position
            subtree_nodes = place_subtree(child, normalize_nd(chosen_direction))
            placed_nodes.append(child)
            placed_nodes.extend(subtree_nodes)

        return placed_nodes

    for root in initial_branch_roots:
        if root not in pos:
            continue
        place_subtree(root, None)

    return local_rejections


def _best_branch_direction_nd(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    parent: int,
    child: int,
    base_direction: np.ndarray,
    edge_length: float,
    max_local_placement_attempts: int,
    dimension: int,
    rng: random.Random,
) -> np.ndarray:
    min_span = 0.35 if dimension == 2 else 0.25
    max_span = 1.35 if dimension == 2 else 1.15
    best_direction = normalize_nd(base_direction)
    best_position = pos[parent] + best_direction * edge_length
    best_clearance = candidate_branch_clearance_nd(
        G=G,
        existing_pos=pos,
        parent=parent,
        child=child,
        child_position=best_position,
    )

    for attempt_idx in range(max_local_placement_attempts):
        angle_budget = _interpolate_attempt_span(
            attempt_idx=attempt_idx,
            max_attempts=max_local_placement_attempts,
            min_span=min_span,
            max_span=max_span,
        )
        direction = sample_branch_direction_nd(
            base_direction=base_direction,
            angle_budget=angle_budget,
            rng=rng,
        )
        position = pos[parent] + normalize_nd(direction) * edge_length
        clearance = candidate_branch_clearance_nd(
            G=G,
            existing_pos=pos,
            parent=parent,
            child=child,
            child_position=position,
        )
        if clearance > best_clearance:
            best_clearance = clearance
            best_direction = normalize_nd(direction)
            best_position = position

    return best_position


def _run_constructive_embedding_2d_with_params(
    G: nx.Graph,
    context: dict[str, Any],
    edge_lengths: dict[tuple[int, int], float],
    embedding_params: dict[str, float],
    max_local_placement_attempts: int,
    max_global_resampling_attempts: int,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    last_pos: dict[int, np.ndarray] = {}
    last_stats: dict[str, float] = {
        "min_nonadjacent_node_distance": 0.0,
        "min_nonincident_edge_distance": 0.0,
        "min_nonincident_node_edge_distance": 0.0,
    }
    total_local_rejections = 0
    last_error = "constructive_embedding_failed"

    for global_attempt in range(1, max_global_resampling_attempts + 1):
        local_rejections = 0
        try:
            pos, cycle_centers, local_rejections = compute_initial_cycle_backbone_geometry(
                G=G,
                cycle_nodes=context["cycle_nodes"],
                cycle_tree_edges=context["cycle_tree_edges"],
                attachments_by_pair=context["attachments_by_pair"],
                edge_lengths=edge_lengths,
                feature_scale=embedding_params["feature_scale"],
                target_packing_distance=embedding_params["target_packing_distance"],
                hard_min_clearance=embedding_params["hard_min_clearance"],
                max_local_placement_attempts=max_local_placement_attempts,
                rng=rng,
            )

            if not pos:
                root = next(iter(G.nodes()))
                pos[root] = np.zeros(2, dtype=float)
                cycle_centers = {}

            local_rejections += place_branch_subtrees_constructively(
                G=G,
                pos=pos,
                branch_children=context["branch_children"],
                branch_parent_map=context["branch_parent_map"],
                branch_depth_map=context["branch_depth_map"],
                initial_branch_roots=context["initial_branch_roots"],
                cycle_nodes=context["cycle_nodes"],
                cycle_centers=cycle_centers,
                backbone_nodes=context["backbone_nodes"],
                edge_lengths=edge_lengths,
                feature_scale=embedding_params["feature_scale"],
                target_packing_distance=embedding_params["target_packing_distance"],
                hard_min_clearance=embedding_params["hard_min_clearance"],
                branch_angle_spread=embedding_params["branch_angle_spread"],
                max_local_placement_attempts=max_local_placement_attempts,
                rng=rng,
            )

            if len(pos) != G.number_of_nodes():
                raise PlacementError("Embedding attempt did not place all graph nodes.")

            embedding_valid, stats = validate_embedding(
                G=G,
                pos=pos,
                target_min_separation=embedding_params["hard_min_clearance"],
                tolerance=0.05 * embedding_params["feature_scale"],
            )
            last_pos = {node: coords.copy() for node, coords in pos.items()}
            last_stats = stats
            total_local_rejections += local_rejections
            if embedding_valid:
                return pos, {
                    "embedding_valid": True,
                    **stats,
                    "global_embedding_attempts": global_attempt,
                    "local_placement_rejections": total_local_rejections,
                    "failure_reason": None,
                }
            last_error = "global_validation_failed"
        except PlacementError as exc:
            total_local_rejections += local_rejections
            last_error = str(exc)
            continue

    return last_pos, {
        "embedding_valid": False,
        **last_stats,
        "global_embedding_attempts": max_global_resampling_attempts,
        "local_placement_rejections": total_local_rejections,
        "failure_reason": last_error,
    }


def _run_constructive_embedding_3d_with_params(
    G: nx.Graph,
    context: dict[str, Any],
    edge_lengths: dict[tuple[int, int], float],
    embedding_params: dict[str, float],
    max_local_placement_attempts: int,
    max_global_resampling_attempts: int,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    last_pos: dict[int, np.ndarray] = {}
    last_stats: dict[str, float] = {
        "min_nonadjacent_node_distance": 0.0,
        "min_nonincident_edge_distance": 0.0,
        "min_nonincident_node_edge_distance": 0.0,
    }
    total_local_rejections = 0
    last_error = "constructive_embedding_failed"

    for global_attempt in range(1, max_global_resampling_attempts + 1):
        local_rejections = 0
        try:
            pos, cycle_centers, local_rejections = compute_initial_cycle_backbone_geometry_3d(
                G=G,
                cycle_nodes=context["cycle_nodes"],
                cycle_tree_edges=context["cycle_tree_edges"],
                attachments_by_pair=context["attachments_by_pair"],
                edge_lengths=edge_lengths,
                feature_scale=embedding_params["feature_scale"],
                target_packing_distance=embedding_params["target_packing_distance"],
                hard_min_clearance=embedding_params["hard_min_clearance"],
                max_local_placement_attempts=max_local_placement_attempts,
                rng=rng,
            )

            if not pos:
                root = next(iter(G.nodes()))
                pos[root] = np.zeros(3, dtype=float)
                cycle_centers = {}

            local_rejections += place_branch_subtrees_constructively_3d(
                G=G,
                pos=pos,
                branch_children=context["branch_children"],
                branch_parent_map=context["branch_parent_map"],
                branch_depth_map=context["branch_depth_map"],
                initial_branch_roots=context["initial_branch_roots"],
                cycle_nodes=context["cycle_nodes"],
                cycle_centers=cycle_centers,
                backbone_nodes=context["backbone_nodes"],
                edge_lengths=edge_lengths,
                feature_scale=embedding_params["feature_scale"],
                target_packing_distance=embedding_params["target_packing_distance"],
                hard_min_clearance=embedding_params["hard_min_clearance"],
                branch_angle_spread=embedding_params["branch_angle_spread"],
                max_local_placement_attempts=max_local_placement_attempts,
                rng=rng,
            )

            if len(pos) != G.number_of_nodes():
                raise PlacementError("Embedding attempt did not place all graph nodes.")

            embedding_valid, stats = validate_embedding_3d(
                G=G,
                pos=pos,
                target_min_separation=embedding_params["hard_min_clearance"],
                tolerance=0.05 * embedding_params["feature_scale"],
            )
            last_pos = {node: coords.copy() for node, coords in pos.items()}
            last_stats = stats
            total_local_rejections += local_rejections
            if embedding_valid:
                return pos, {
                    "embedding_valid": True,
                    **stats,
                    "global_embedding_attempts": global_attempt,
                    "local_placement_rejections": total_local_rejections,
                    "failure_reason": None,
                }
            last_error = "global_validation_failed"
        except PlacementError as exc:
            total_local_rejections += local_rejections
            last_error = str(exc)
            continue

    return last_pos, {
        "embedding_valid": False,
        **last_stats,
        "global_embedding_attempts": max_global_resampling_attempts,
        "local_placement_rejections": total_local_rejections,
        "failure_reason": last_error,
    }


def sample_embedding_params(
    feature_scale_range: tuple[float, float],
    separation_to_feature_ratio_range: tuple[float, float],
    packing_to_feature_ratio_range: tuple[float, float] | None,
    noise_to_feature_ratio_range: tuple[float, float],
    sampling_density_per_feature_range: tuple[float, float],
    edge_length_jitter_ratio_range: tuple[float, float],
    branch_angle_spread_range: tuple[float, float],
    hard_min_clearance_to_feature_ratio: float,
    rng: random.Random,
) -> dict[str, float]:
    """Sample per-instance embedding difficulty parameters."""

    feature_scale = rng.uniform(feature_scale_range[0], feature_scale_range[1])
    sampled_packing_range = (
        packing_to_feature_ratio_range
        if packing_to_feature_ratio_range is not None
        else separation_to_feature_ratio_range
    )
    packing_ratio = rng.uniform(sampled_packing_range[0], sampled_packing_range[1])
    noise_ratio = rng.uniform(
        noise_to_feature_ratio_range[0],
        noise_to_feature_ratio_range[1],
    )
    sampling_density = rng.uniform(
        sampling_density_per_feature_range[0],
        sampling_density_per_feature_range[1],
    )
    edge_length_jitter_ratio = rng.uniform(
        edge_length_jitter_ratio_range[0],
        edge_length_jitter_ratio_range[1],
    )
    branch_angle_spread = rng.uniform(
        branch_angle_spread_range[0],
        branch_angle_spread_range[1],
    )
    hard_min_clearance = hard_min_clearance_to_feature_ratio * feature_scale
    preferred_min_separation = max(
        hard_min_clearance,
        min(0.8 * packing_ratio * feature_scale, 0.95 * feature_scale),
    )
    return {
        "feature_scale": feature_scale,
        "packing_ratio": packing_ratio,
        "separation_ratio": packing_ratio,
        "target_packing_distance": packing_ratio * feature_scale,
        "hard_min_clearance_ratio": hard_min_clearance_to_feature_ratio,
        "hard_min_clearance": hard_min_clearance,
        "target_min_separation": hard_min_clearance,
        "preferred_min_separation": preferred_min_separation,
        "preferred_min_separation_ratio": preferred_min_separation / max(feature_scale, 1e-9),
        "noise_ratio": noise_ratio,
        "sampling_density_per_feature": sampling_density,
        "edge_length_jitter_ratio": edge_length_jitter_ratio,
        "branch_angle_spread": branch_angle_spread,
    }


def assign_target_edge_lengths(
    G: nx.Graph,
    feature_scale: float,
    jitter_ratio: float,
    rng: random.Random,
) -> dict[tuple[int, int], float]:
    """Assign a target geometric length to every graph edge."""

    lengths: dict[tuple[int, int], float] = {}
    for u, v in sorted(G.edges()):
        jitter = rng.uniform(-jitter_ratio, jitter_ratio)
        lengths[_edge_key(u, v)] = feature_scale * (1.0 + jitter)
    return lengths


def compute_initial_cycle_backbone_geometry(
    G: nx.Graph,
    cycle_nodes: dict[int, list[int]],
    cycle_tree_edges: list[tuple[int, int]],
    attachments_by_pair: dict[tuple[int, int], dict[str, Any]],
    edge_lengths: dict[tuple[int, int], float],
    feature_scale: float,
    target_packing_distance: float,
    hard_min_clearance: float,
    max_local_placement_attempts: int,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]:
    """Constructively place the cycle backbone before any branches."""

    if not cycle_nodes:
        return {}, {}, 0

    pos: dict[int, np.ndarray] = {}
    cycle_centers: dict[int, np.ndarray] = {}
    local_rejections = 0

    root_cycle = min(cycle_nodes)
    root_nodes = cycle_nodes[root_cycle]
    root_edge_lengths = cycle_edge_lengths(root_nodes, edge_lengths)
    root_positions = place_cycle_polygon(
        cycle_nodes=root_nodes,
        edge_lengths=root_edge_lengths,
        center=np.zeros(2, dtype=float),
        rotation=rng.uniform(0.0, 2.0 * math.pi),
    )
    pos.update(root_positions)
    cycle_centers[root_cycle] = np.mean(np.stack([pos[node] for node in root_nodes]), axis=0)

    tree_adj = _tree_adjacency(len(cycle_nodes), cycle_tree_edges)
    seen = {root_cycle}
    traversal = deque([root_cycle])

    while traversal:
        parent_cycle = traversal.popleft()
        parent_center = cycle_centers[parent_cycle]
        for child_cycle in tree_adj.get(parent_cycle, []):
            if child_cycle in seen:
                continue
            seen.add(child_cycle)
            traversal.append(child_cycle)

            pair = tuple(sorted((parent_cycle, child_cycle)))
            attachment = attachments_by_pair.get(pair)
            if attachment is None:
                raise PlacementError(f"Missing attachment metadata for cycle pair {pair}.")

            child_nodes = cycle_nodes[child_cycle]
            placed = False
            for _ in range(max_local_placement_attempts):
                if attachment["attachment_type"] == "path":
                    candidate_pos, child_center = _construct_path_attached_cycle(
                        cycle_nodes=child_nodes,
                        attachment=attachment,
                        existing_pos=pos,
                        parent_center=parent_center,
                        edge_lengths=edge_lengths,
                        target_packing_distance=target_packing_distance,
                        rng=rng,
                    )
                elif attachment["attachment_type"] == "shared_vertex":
                    candidate_pos, child_center = _construct_shared_vertex_cycle(
                        cycle_nodes=child_nodes,
                        attachment=attachment,
                        existing_pos=pos,
                        parent_center=parent_center,
                        edge_lengths=edge_lengths,
                        target_packing_distance=target_packing_distance,
                        rng=rng,
                    )
                elif attachment["attachment_type"] == "shared_path":
                    candidate_pos, child_center = _construct_shared_path_cycle(
                        cycle_nodes=child_nodes,
                        attachment=attachment,
                        existing_pos=pos,
                        parent_center=parent_center,
                        edge_lengths=edge_lengths,
                        feature_scale=feature_scale,
                        target_packing_distance=target_packing_distance,
                        rng=rng,
                    )
                else:
                    raise PlacementError(
                        f"Unsupported attachment type {attachment['attachment_type']}."
                    )

                if not validate_partial_embedding(
                    G=G,
                    existing_pos=pos,
                    candidate_pos=candidate_pos,
                    target_min_separation=hard_min_clearance,
                    tolerance_ratio=0.15,
                ):
                    local_rejections += 1
                    continue

                pos.update(candidate_pos)
                cycle_centers[child_cycle] = child_center
                placed = True
                break

            if not placed:
                raise PlacementError(f"Failed to place child cycle {child_cycle}.")

    return pos, cycle_centers, local_rejections


def _construct_path_attached_cycle(
    cycle_nodes: list[int],
    attachment: dict[str, Any],
    existing_pos: dict[int, np.ndarray],
    parent_center: np.ndarray,
    edge_lengths: dict[tuple[int, int], float],
    target_packing_distance: float,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Construct candidate placement for a path-separated child cycle."""

    path_nodes = list(attachment["path_nodes"])
    anchor_a = attachment["anchor_a"]
    anchor_b = attachment["anchor_b"]
    if anchor_a not in existing_pos:
        raise PlacementError("Path attachment anchor on parent cycle is not placed.")

    child_edge_lengths = cycle_edge_lengths(cycle_nodes, edge_lengths)
    child_radius = polygon_radius_for_cycle(child_edge_lengths)

    direction = _normalize(existing_pos[anchor_a] - parent_center)
    if np.linalg.norm(direction) < 1e-10:
        direction = _random_unit_vector(rng)
    direction = _rotate(direction, rng.uniform(-0.9, 0.9))

    path_edge_lengths = [
        edge_lengths[_edge_key(path_nodes[idx], path_nodes[idx + 1])]
        for idx in range(len(path_nodes) - 1)
    ]
    total_path_length = sum(path_edge_lengths)
    child_center = existing_pos[anchor_a] + direction * (
        total_path_length
        + child_radius
        + _cycle_packing_offset(
            target_packing_distance=target_packing_distance,
            feature_radius=child_radius,
        )
    )
    anchor_b_pos = existing_pos[anchor_a] + direction * total_path_length

    path_positions = place_polyline_path(
        path_nodes=path_nodes,
        edge_lengths=path_edge_lengths,
        start_position=existing_pos[anchor_a],
        direction=direction,
        rng=rng,
        angular_jitter=0.0,
    )
    path_positions[anchor_b] = anchor_b_pos

    cycle_positions = place_cycle_polygon_from_anchor(
        cycle_nodes=cycle_nodes,
        edge_lengths=child_edge_lengths,
        anchor_node=anchor_b,
        anchor_position=anchor_b_pos,
        center=child_center,
    )

    candidate_pos = {
        node: coords
        for node, coords in {**path_positions, **cycle_positions}.items()
        if node not in existing_pos
    }
    return candidate_pos, child_center


def _construct_shared_vertex_cycle(
    cycle_nodes: list[int],
    attachment: dict[str, Any],
    existing_pos: dict[int, np.ndarray],
    parent_center: np.ndarray,
    edge_lengths: dict[tuple[int, int], float],
    target_packing_distance: float,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Construct candidate placement for a child cycle sharing one vertex."""

    shared_node = attachment["shared_node"]
    if shared_node not in existing_pos:
        raise PlacementError("Shared vertex is not yet placed.")

    child_edge_lengths = cycle_edge_lengths(cycle_nodes, edge_lengths)
    child_radius = polygon_radius_for_cycle(child_edge_lengths)
    direction = _normalize(existing_pos[shared_node] - parent_center)
    if np.linalg.norm(direction) < 1e-10:
        direction = _random_unit_vector(rng)
    direction = _rotate(direction, rng.uniform(-1.0, 1.0))
    child_center = existing_pos[shared_node] + direction * (
        child_radius
        + _cycle_packing_offset(
            target_packing_distance=target_packing_distance,
            feature_radius=child_radius,
        )
    )

    cycle_positions = place_cycle_polygon_from_anchor(
        cycle_nodes=cycle_nodes,
        edge_lengths=child_edge_lengths,
        anchor_node=shared_node,
        anchor_position=existing_pos[shared_node],
        center=child_center,
    )
    candidate_pos = {
        node: coords for node, coords in cycle_positions.items() if node not in existing_pos
    }
    return candidate_pos, child_center


def _construct_shared_path_cycle(
    cycle_nodes: list[int],
    attachment: dict[str, Any],
    existing_pos: dict[int, np.ndarray],
    parent_center: np.ndarray,
    edge_lengths: dict[tuple[int, int], float],
    feature_scale: float,
    target_packing_distance: float,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Construct candidate placement for a child cycle sharing a path."""

    shared_nodes = list(attachment["shared_path_nodes"])
    for node in shared_nodes:
        if node not in existing_pos:
            raise PlacementError("Shared-path anchor nodes are not fully placed.")

    shared_in_cycle = _locate_path_in_cycle(cycle_nodes, shared_nodes)
    complement_path = _complementary_cycle_path(cycle_nodes, shared_in_cycle)
    edge_seq = [
        edge_lengths[_edge_key(complement_path[idx], complement_path[idx + 1])]
        for idx in range(len(complement_path) - 1)
    ]

    start_pos = existing_pos[complement_path[0]]
    end_pos = existing_pos[complement_path[-1]]
    midpoint = 0.5 * (start_pos + end_pos)
    preferred_normal = _perpendicular(_normalize(end_pos - start_pos))
    away_from_parent = midpoint - parent_center
    if float(np.dot(preferred_normal, away_from_parent)) < 0.0:
        preferred_normal = -preferred_normal
    preferred_normal = _rotate(preferred_normal, rng.uniform(-0.35, 0.35))

    complement_positions = place_polyline_between_anchors(
        path_nodes=complement_path,
        edge_lengths=edge_seq,
        start_position=start_pos,
        end_position=end_pos,
        preferred_normal=preferred_normal,
        feature_scale=feature_scale,
        rng=rng,
    )
    _inflate_complement_positions_2d(
        complement_positions=complement_positions,
        complement_path=complement_path,
        preferred_normal=preferred_normal,
        edge_seq=edge_seq,
        target_packing_distance=target_packing_distance,
    )

    merged = dict(existing_pos)
    merged.update(complement_positions)
    child_center = np.mean(np.stack([merged[node] for node in cycle_nodes]), axis=0)
    return complement_positions, child_center


def compute_initial_cycle_backbone_geometry_3d(
    G: nx.Graph,
    cycle_nodes: dict[int, list[int]],
    cycle_tree_edges: list[tuple[int, int]],
    attachments_by_pair: dict[tuple[int, int], dict[str, Any]],
    edge_lengths: dict[tuple[int, int], float],
    feature_scale: float,
    target_packing_distance: float,
    hard_min_clearance: float,
    max_local_placement_attempts: int,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]:
    """Constructively place the cycle backbone in 3D before any branches."""

    if not cycle_nodes:
        return {}, {}, 0

    pos: dict[int, np.ndarray] = {}
    cycle_centers: dict[int, np.ndarray] = {}
    local_rejections = 0

    root_cycle = min(cycle_nodes)
    root_nodes = cycle_nodes[root_cycle]
    root_edge_lengths = cycle_edge_lengths(root_nodes, edge_lengths)
    root_positions = place_cycle_polygon_3d(
        cycle_nodes=root_nodes,
        edge_lengths=root_edge_lengths,
        center=np.zeros(3, dtype=float),
        normal=random_unit_vector_3d(rng),
        rotation=rng.uniform(0.0, 2.0 * math.pi),
    )
    pos.update(root_positions)
    cycle_centers[root_cycle] = np.mean(np.stack([pos[node] for node in root_nodes]), axis=0)

    tree_adj = _tree_adjacency(len(cycle_nodes), cycle_tree_edges)
    seen = {root_cycle}
    traversal = deque([root_cycle])

    while traversal:
        parent_cycle = traversal.popleft()
        parent_center = cycle_centers[parent_cycle]
        for child_cycle in tree_adj.get(parent_cycle, []):
            if child_cycle in seen:
                continue
            seen.add(child_cycle)
            traversal.append(child_cycle)

            pair = tuple(sorted((parent_cycle, child_cycle)))
            attachment = attachments_by_pair.get(pair)
            if attachment is None:
                raise PlacementError(f"Missing attachment metadata for cycle pair {pair}.")

            child_nodes = cycle_nodes[child_cycle]
            placed = False
            for _ in range(max_local_placement_attempts):
                if attachment["attachment_type"] == "path":
                    candidate_pos, child_center = _construct_path_attached_cycle_3d(
                        cycle_nodes=child_nodes,
                        attachment=attachment,
                        existing_pos=pos,
                        parent_center=parent_center,
                        edge_lengths=edge_lengths,
                        target_packing_distance=target_packing_distance,
                        rng=rng,
                    )
                elif attachment["attachment_type"] == "shared_vertex":
                    candidate_pos, child_center = _construct_shared_vertex_cycle_3d(
                        cycle_nodes=child_nodes,
                        attachment=attachment,
                        existing_pos=pos,
                        parent_center=parent_center,
                        edge_lengths=edge_lengths,
                        target_packing_distance=target_packing_distance,
                        rng=rng,
                    )
                elif attachment["attachment_type"] == "shared_path":
                    candidate_pos, child_center = _construct_shared_path_cycle_3d(
                        cycle_nodes=child_nodes,
                        attachment=attachment,
                        existing_pos=pos,
                        parent_center=parent_center,
                        edge_lengths=edge_lengths,
                        feature_scale=feature_scale,
                        target_packing_distance=target_packing_distance,
                        rng=rng,
                    )
                else:
                    raise PlacementError(
                        f"Unsupported attachment type {attachment['attachment_type']}."
                    )

                if not validate_partial_embedding_3d(
                    G=G,
                    existing_pos=pos,
                    candidate_pos=candidate_pos,
                    target_min_separation=hard_min_clearance,
                    tolerance_ratio=0.15,
                ):
                    local_rejections += 1
                    continue

                pos.update(candidate_pos)
                cycle_centers[child_cycle] = child_center
                placed = True
                break

            if not placed:
                raise PlacementError(f"Failed to place child cycle {child_cycle}.")

    return pos, cycle_centers, local_rejections


def _construct_path_attached_cycle_3d(
    cycle_nodes: list[int],
    attachment: dict[str, Any],
    existing_pos: dict[int, np.ndarray],
    parent_center: np.ndarray,
    edge_lengths: dict[tuple[int, int], float],
    target_packing_distance: float,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Construct candidate placement for a path-separated child cycle in 3D."""

    path_nodes = list(attachment["path_nodes"])
    anchor_a = attachment["anchor_a"]
    anchor_b = attachment["anchor_b"]
    if anchor_a not in existing_pos:
        raise PlacementError("Path attachment anchor on parent cycle is not placed.")

    child_edge_lengths = cycle_edge_lengths(cycle_nodes, edge_lengths)
    child_radius = polygon_radius_for_cycle(child_edge_lengths)
    direction = existing_pos[anchor_a] - parent_center
    if np.linalg.norm(direction) < 1e-10:
        direction = random_unit_vector_3d(rng)
    direction = _normalize_3d(direction + 0.35 * random_unit_vector_3d(rng))

    path_edge_lengths = [
        edge_lengths[_edge_key(path_nodes[idx], path_nodes[idx + 1])]
        for idx in range(len(path_nodes) - 1)
    ]
    total_path_length = sum(path_edge_lengths)
    child_center = existing_pos[anchor_a] + direction * (
        total_path_length
        + child_radius
        + _cycle_packing_offset(
            target_packing_distance=target_packing_distance,
            feature_radius=child_radius,
        )
    )
    anchor_b_pos = existing_pos[anchor_a] + direction * total_path_length
    plane_normal = np.cross(direction, random_unit_vector_3d(rng))
    if np.linalg.norm(plane_normal) < 1e-10:
        plane_normal = random_unit_vector_3d(rng)

    path_positions = place_polyline_path_3d(
        path_nodes=path_nodes,
        edge_lengths=path_edge_lengths,
        start_position=existing_pos[anchor_a],
        direction=direction,
        rng=rng,
        angular_jitter=0.0,
    )
    path_positions[anchor_b] = anchor_b_pos

    cycle_positions = place_cycle_polygon_from_anchor_3d(
        cycle_nodes=cycle_nodes,
        edge_lengths=child_edge_lengths,
        anchor_node=anchor_b,
        anchor_position=anchor_b_pos,
        center=child_center,
        plane_normal=plane_normal,
    )
    candidate_pos = {
        node: coords
        for node, coords in {**path_positions, **cycle_positions}.items()
        if node not in existing_pos
    }
    return candidate_pos, child_center


def _construct_shared_vertex_cycle_3d(
    cycle_nodes: list[int],
    attachment: dict[str, Any],
    existing_pos: dict[int, np.ndarray],
    parent_center: np.ndarray,
    edge_lengths: dict[tuple[int, int], float],
    target_packing_distance: float,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Construct candidate placement for a child cycle sharing one vertex in 3D."""

    shared_node = attachment["shared_node"]
    if shared_node not in existing_pos:
        raise PlacementError("Shared vertex is not yet placed.")

    child_edge_lengths = cycle_edge_lengths(cycle_nodes, edge_lengths)
    child_radius = polygon_radius_for_cycle(child_edge_lengths)
    direction = existing_pos[shared_node] - parent_center
    if np.linalg.norm(direction) < 1e-10:
        direction = random_unit_vector_3d(rng)
    direction = _normalize_3d(direction + 0.35 * random_unit_vector_3d(rng))
    child_center = existing_pos[shared_node] + direction * (
        child_radius
        + _cycle_packing_offset(
            target_packing_distance=target_packing_distance,
            feature_radius=child_radius,
        )
    )
    plane_normal = np.cross(direction, random_unit_vector_3d(rng))
    if np.linalg.norm(plane_normal) < 1e-10:
        plane_normal = random_unit_vector_3d(rng)

    cycle_positions = place_cycle_polygon_from_anchor_3d(
        cycle_nodes=cycle_nodes,
        edge_lengths=child_edge_lengths,
        anchor_node=shared_node,
        anchor_position=existing_pos[shared_node],
        center=child_center,
        plane_normal=plane_normal,
    )
    candidate_pos = {
        node: coords for node, coords in cycle_positions.items() if node not in existing_pos
    }
    return candidate_pos, child_center


def _construct_shared_path_cycle_3d(
    cycle_nodes: list[int],
    attachment: dict[str, Any],
    existing_pos: dict[int, np.ndarray],
    parent_center: np.ndarray,
    edge_lengths: dict[tuple[int, int], float],
    feature_scale: float,
    target_packing_distance: float,
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Construct candidate placement for a child cycle sharing a path in 3D."""

    shared_nodes = list(attachment["shared_path_nodes"])
    for node in shared_nodes:
        if node not in existing_pos:
            raise PlacementError("Shared-path anchor nodes are not fully placed.")

    shared_in_cycle = _locate_path_in_cycle(cycle_nodes, shared_nodes)
    complement_path = _complementary_cycle_path(cycle_nodes, shared_in_cycle)
    edge_seq = [
        edge_lengths[_edge_key(complement_path[idx], complement_path[idx + 1])]
        for idx in range(len(complement_path) - 1)
    ]

    start_pos = existing_pos[complement_path[0]]
    end_pos = existing_pos[complement_path[-1]]
    midpoint = 0.5 * (start_pos + end_pos)
    chord = end_pos - start_pos
    preferred_normal = midpoint - parent_center
    preferred_normal = preferred_normal - float(np.dot(preferred_normal, chord)) * _normalize_3d(
        chord if np.linalg.norm(chord) > 1e-12 else random_unit_vector_3d(rng)
    )
    if np.linalg.norm(preferred_normal) < 1e-10:
        preferred_normal = random_unit_vector_3d(rng)

    complement_positions = place_polyline_between_anchors_3d(
        path_nodes=complement_path,
        edge_lengths=edge_seq,
        start_position=start_pos,
        end_position=end_pos,
        preferred_normal=preferred_normal,
        feature_scale=feature_scale,
        rng=rng,
    )
    _inflate_complement_positions_3d(
        complement_positions=complement_positions,
        complement_path=complement_path,
        preferred_normal=preferred_normal,
        edge_seq=edge_seq,
        target_packing_distance=target_packing_distance,
    )

    merged = dict(existing_pos)
    merged.update(complement_positions)
    child_center = np.mean(np.stack([merged[node] for node in cycle_nodes]), axis=0)
    return complement_positions, child_center


def place_branch_subtrees_constructively(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    branch_children: dict[int, list[int]],
    branch_parent_map: dict[int, int],
    branch_depth_map: dict[int, int],
    initial_branch_roots: list[int],
    cycle_nodes: dict[int, list[int]],
    cycle_centers: dict[int, np.ndarray],
    backbone_nodes: set[int],
    edge_lengths: dict[tuple[int, int], float],
    feature_scale: float,
    target_packing_distance: float,
    hard_min_clearance: float,
    branch_angle_spread: float,
    max_local_placement_attempts: int,
    rng: random.Random,
) -> int:
    """Recursively place branch trees attached to the backbone."""

    del branch_parent_map
    if not branch_children:
        return 0

    node_cycle_membership = build_node_cycle_membership(cycle_nodes)
    backbone_centroid = np.mean(np.stack([pos[node] for node in backbone_nodes if node in pos]), axis=0)
    local_rejections = 0

    def place_subtree(parent: int, incoming_direction: np.ndarray | None) -> list[int]:
        nonlocal local_rejections
        children = list(branch_children.get(parent, []))
        if not children:
            return []

        ordered_children = _order_branch_children(children, parent, branch_depth_map)
        preferred_direction = choose_outward_direction(
            node=parent,
            pos=pos,
            cycle_membership=node_cycle_membership,
            cycle_centers=cycle_centers,
            backbone_centroid=backbone_centroid,
            incoming_direction=incoming_direction,
            rng=rng,
        )
        child_directions = angularly_spread_child_directions(
            preferred_direction=preferred_direction,
            num_children=len(ordered_children),
            spread=_effective_branch_spread(
                branch_angle_spread=branch_angle_spread,
                target_packing_distance=target_packing_distance,
                feature_scale=feature_scale,
            ),
            rng=rng,
            continuation_child=ordered_children[0] if ordered_children else None,
            parent=parent,
            branch_depth_map=branch_depth_map,
        )

        placed_nodes: list[int] = []
        for child, base_direction in zip(ordered_children, child_directions):
            edge_length = edge_lengths[_edge_key(parent, child)]
            chosen_direction = None
            chosen_position = None

            for attempt_idx in range(max_local_placement_attempts):
                angle_budget = _interpolate_attempt_span(
                    attempt_idx=attempt_idx,
                    max_attempts=max_local_placement_attempts,
                    min_span=0.35,
                    max_span=1.35,
                )
                direction = _rotate(base_direction, rng.uniform(-angle_budget, angle_budget))
                candidate_position = pos[parent] + _normalize(direction) * edge_length
                candidate_pos = {child: candidate_position}

                if validate_partial_embedding(
                    G=G,
                    existing_pos=pos,
                    candidate_pos=candidate_pos,
                    target_min_separation=hard_min_clearance,
                    tolerance_ratio=0.15,
                ):
                    chosen_direction = direction
                    chosen_position = candidate_position
                    break
                local_rejections += 1

            if chosen_direction is None or chosen_position is None:
                best_direction, best_position = _best_branch_direction(
                    G=G,
                    pos=pos,
                    parent=parent,
                    child=child,
                    base_direction=base_direction,
                    edge_length=edge_length,
                    max_local_placement_attempts=max_local_placement_attempts,
                    rng=rng,
                )
                raise PlacementError(
                    f"Failed to place branch child {child} from parent {parent}; "
                    f"best clearance={candidate_branch_clearance(G, pos, parent, child, best_position):.3f}."
                )

            pos[child] = chosen_position
            subtree_nodes = place_subtree(child, _normalize(chosen_direction))
            placed_nodes.append(child)
            placed_nodes.extend(subtree_nodes)

        return placed_nodes

    for root in initial_branch_roots:
        if root not in pos:
            continue
        place_subtree(root, None)

    return local_rejections


def place_branch_subtrees_constructively_3d(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    branch_children: dict[int, list[int]],
    branch_parent_map: dict[int, int],
    branch_depth_map: dict[int, int],
    initial_branch_roots: list[int],
    cycle_nodes: dict[int, list[int]],
    cycle_centers: dict[int, np.ndarray],
    backbone_nodes: set[int],
    edge_lengths: dict[tuple[int, int], float],
    feature_scale: float,
    target_packing_distance: float,
    hard_min_clearance: float,
    branch_angle_spread: float,
    max_local_placement_attempts: int,
    rng: random.Random,
) -> int:
    """Recursively place branch trees attached to the backbone in 3D."""

    del branch_parent_map
    if not branch_children:
        return 0

    node_cycle_membership = build_node_cycle_membership(cycle_nodes)
    backbone_positions = [pos[node] for node in backbone_nodes if node in pos]
    if backbone_positions:
        backbone_centroid = np.mean(np.stack(backbone_positions), axis=0)
    else:
        backbone_centroid = np.zeros(3, dtype=float)
    local_rejections = 0

    def place_subtree(parent: int, incoming_direction: np.ndarray | None) -> list[int]:
        nonlocal local_rejections
        children = list(branch_children.get(parent, []))
        if not children:
            return []

        ordered_children = _order_branch_children(children, parent, branch_depth_map)
        preferred_direction = choose_outward_direction_3d(
            node=parent,
            pos=pos,
            cycle_membership=node_cycle_membership,
            cycle_centers=cycle_centers,
            backbone_centroid=backbone_centroid,
            incoming_direction=incoming_direction,
            rng=rng,
        )
        child_directions = spread_child_directions_3d(
            preferred_direction=preferred_direction,
            num_children=len(ordered_children),
            spread=_effective_branch_spread(
                branch_angle_spread=branch_angle_spread,
                target_packing_distance=target_packing_distance,
                feature_scale=feature_scale,
            ),
            rng=rng,
            continuation_child=ordered_children[0] if ordered_children else None,
            parent=parent,
            branch_depth_map=branch_depth_map,
        )

        placed_nodes: list[int] = []
        for child, base_direction in zip(ordered_children, child_directions):
            edge_length = edge_lengths[_edge_key(parent, child)]
            chosen_direction = None
            chosen_position = None

            for attempt_idx in range(max_local_placement_attempts):
                angle_budget = _interpolate_attempt_span(
                    attempt_idx=attempt_idx,
                    max_attempts=max_local_placement_attempts,
                    min_span=0.25,
                    max_span=1.15,
                )
                direction = _sample_branch_direction_3d(
                    base_direction=base_direction,
                    angle_budget=angle_budget,
                    rng=rng,
                )
                candidate_position = pos[parent] + _normalize_3d(direction) * edge_length
                candidate_pos = {child: candidate_position}

                if validate_partial_embedding_3d(
                    G=G,
                    existing_pos=pos,
                    candidate_pos=candidate_pos,
                    target_min_separation=hard_min_clearance,
                    tolerance_ratio=0.15,
                ):
                    chosen_direction = direction
                    chosen_position = candidate_position
                    break
                local_rejections += 1

            if chosen_direction is None or chosen_position is None:
                best_position = _best_branch_direction_3d(
                    G=G,
                    pos=pos,
                    parent=parent,
                    child=child,
                    base_direction=base_direction,
                    edge_length=edge_length,
                    max_local_placement_attempts=max_local_placement_attempts,
                    rng=rng,
                )
                raise PlacementError(
                    f"Failed to place branch child {child} from parent {parent}; "
                    f"best clearance={candidate_branch_clearance_3d(G, pos, parent, child, best_position):.3f}."
                )

            pos[child] = chosen_position
            subtree_nodes = place_subtree(child, _normalize_3d(chosen_direction))
            placed_nodes.append(child)
            placed_nodes.extend(subtree_nodes)

        return placed_nodes

    for root in initial_branch_roots:
        if root not in pos:
            continue
        place_subtree(root, None)

    return local_rejections


def fallback_planar_embedding(
    G: nx.Graph,
    feature_scale: float,
    target_min_separation: float,
) -> tuple[dict[int, np.ndarray], bool, dict[str, float]]:
    """Fallback helper based on a planar layout, used only after constructive failure."""

    try:
        layout = nx.planar_layout(G)
    except Exception:
        return {}, False, {
            "min_nonadjacent_node_distance": 0.0,
            "min_nonincident_edge_distance": 0.0,
            "min_nonincident_node_edge_distance": 0.0,
        }

    pos = {node: np.asarray(coords, dtype=float) for node, coords in layout.items()}
    edge_lengths = [
        float(np.linalg.norm(pos[u] - pos[v]))
        for u, v in G.edges()
        if u in pos and v in pos
    ]
    median_edge = float(np.median(edge_lengths)) if edge_lengths else 1.0
    scale = feature_scale / max(median_edge, 1e-9)
    pos = {node: coords * scale for node, coords in pos.items()}

    _, stats = validate_embedding(
        G=G,
        pos=pos,
        target_min_separation=0.0,
        tolerance=0.0,
    )
    finite_values = [
        value
        for value in (
            stats["min_nonadjacent_node_distance"],
            stats["min_nonincident_edge_distance"],
            stats["min_nonincident_node_edge_distance"],
        )
        if math.isfinite(value) and value > 0.0
    ]
    if finite_values:
        current_min = min(finite_values)
        if current_min < target_min_separation:
            sep_scale = target_min_separation / current_min
            pos = {node: coords * sep_scale for node, coords in pos.items()}

    valid, stats = validate_embedding(
        G=G,
        pos=pos,
        target_min_separation=target_min_separation,
        tolerance=0.05 * feature_scale,
    )
    return pos, valid, stats


def fallback_spring_embedding(
    G: nx.Graph,
    feature_scale: float,
    target_min_separation: float,
    seed: int | None = None,
) -> tuple[dict[int, np.ndarray], bool, dict[str, float]]:
    """Fallback helper based on a force-directed layout."""

    layout = nx.spring_layout(
        G,
        seed=seed,
        iterations=300,
        k=2.5 * feature_scale,
    )
    pos = {node: np.asarray(coords, dtype=float) for node, coords in layout.items()}

    edge_lengths = [
        float(np.linalg.norm(pos[u] - pos[v]))
        for u, v in G.edges()
        if u in pos and v in pos
    ]
    median_edge = float(np.median(edge_lengths)) if edge_lengths else 1.0
    scale = feature_scale / max(median_edge, 1e-9)
    pos = {node: coords * scale for node, coords in pos.items()}

    _, stats = validate_embedding(
        G=G,
        pos=pos,
        target_min_separation=0.0,
        tolerance=0.0,
    )
    finite_values = [
        value
        for value in (
            stats["min_nonadjacent_node_distance"],
            stats["min_nonincident_edge_distance"],
            stats["min_nonincident_node_edge_distance"],
        )
        if math.isfinite(value) and value > 0.0
    ]
    if finite_values:
        current_min = min(finite_values)
        if current_min < target_min_separation:
            sep_scale = target_min_separation / current_min
            pos = {node: coords * sep_scale for node, coords in pos.items()}

    valid, stats = validate_embedding(
        G=G,
        pos=pos,
        target_min_separation=target_min_separation,
        tolerance=0.05 * feature_scale,
    )
    return pos, valid, stats


def _locate_path_in_cycle(cycle_nodes: list[int], path_nodes: list[int]) -> list[int]:
    n = len(cycle_nodes)
    path_len = len(path_nodes)
    for start in range(n):
        candidate = [cycle_nodes[(start + offset) % n] for offset in range(path_len)]
        if candidate == path_nodes:
            return candidate
        if candidate == list(reversed(path_nodes)):
            return candidate
    raise PlacementError("Failed to locate shared path in cycle ordering.")


def _complementary_cycle_path(cycle_nodes: list[int], shared_path_nodes: list[int]) -> list[int]:
    n = len(cycle_nodes)
    path_len = len(shared_path_nodes)
    for start in range(n):
        candidate = [cycle_nodes[(start + offset) % n] for offset in range(path_len)]
        if candidate == shared_path_nodes:
            complement = [candidate[-1]]
            cursor = (start + path_len - 1) % n
            for _ in range(n - path_len):
                cursor = (cursor + 1) % n
                complement.append(cycle_nodes[cursor])
            complement.append(candidate[0])
            return complement
    raise PlacementError("Failed to construct complementary cycle path.")


def _tree_adjacency(
    num_cycles: int,
    tree_edges: list[tuple[int, int]],
) -> dict[int, list[int]]:
    adj = {cycle_id: [] for cycle_id in range(num_cycles)}
    for i, j in tree_edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _cycle_packing_offset(
    target_packing_distance: float,
    feature_radius: float,
) -> float:
    """Return an outward motif offset that grows with the requested packing scale."""

    return max(0.0, 0.7 * target_packing_distance - 0.2 * feature_radius)


def _effective_branch_spread(
    branch_angle_spread: float,
    target_packing_distance: float,
    feature_scale: float,
) -> float:
    """Shrink sibling branch fanout for crowded layouts and widen it for spacious ones."""

    packing_ratio = target_packing_distance / max(feature_scale, 1e-9)
    spread_scale = min(1.45, max(0.35, 0.15 + 0.75 * packing_ratio))
    return branch_angle_spread * spread_scale


def _inflate_complement_positions_2d(
    complement_positions: dict[int, np.ndarray],
    complement_path: list[int],
    preferred_normal: np.ndarray,
    edge_seq: list[float],
    target_packing_distance: float,
) -> None:
    """Bow complementary path vertices outward to realize the requested packing."""

    if len(complement_path) <= 2 or not edge_seq:
        return

    normal = _normalize(preferred_normal)
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


def _inflate_complement_positions_3d(
    complement_positions: dict[int, np.ndarray],
    complement_path: list[int],
    preferred_normal: np.ndarray,
    edge_seq: list[float],
    target_packing_distance: float,
) -> None:
    """Bow complementary path vertices outward in 3D to realize the requested packing."""

    if len(complement_path) <= 2 or not edge_seq:
        return

    normal = _normalize_3d(preferred_normal)
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


def _compute_embedding_stats(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    dimension: int,
) -> dict[str, float]:
    """Return geometric separation stats for an embedding without enforcing validity."""

    del dimension
    _, stats = validate_embedding_nd(
        G=G,
        pos=pos,
        target_min_separation=0.0,
        tolerance=0.0,
    )
    return stats


def _compute_edge_length_summary(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
) -> dict[str, float]:
    """Summarize embedded edge lengths."""

    edge_lengths = [
        float(np.linalg.norm(np.asarray(pos[u], dtype=float) - np.asarray(pos[v], dtype=float)))
        for u, v in G.edges()
        if u in pos and v in pos
    ]
    if not edge_lengths:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    values = np.asarray(edge_lengths, dtype=float)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _min_structural_separation(stats: dict[str, float]) -> float:
    """Collapse the structural separation stats into one realized minimum distance."""

    candidates = [
        float(stats["min_nonadjacent_node_distance"]),
        float(stats["min_nonincident_edge_distance"]),
        float(stats["min_nonincident_node_edge_distance"]),
    ]
    finite_candidates = [value for value in candidates if math.isfinite(value)]
    if not finite_candidates:
        return math.inf
    return min(finite_candidates)


def _is_crowding_embedding_admissible(
    metadata: dict[str, Any],
    epsilon_sep: float,
) -> bool:
    """Return whether a crowded embedding respects the structural separation floor."""

    return (
        float(metadata["realized_post_crowding_min_nonadjacent_node_distance"]) >= epsilon_sep
        and float(metadata["realized_post_crowding_min_nonincident_edge_distance"]) >= epsilon_sep
        and float(metadata["realized_post_crowding_min_nonincident_node_edge_distance"]) >= epsilon_sep
    )


def _extract_feature_scale_from_metadata(metadata: dict[str, Any]) -> float:
    embedding_params = metadata.get("embedding_params")
    if isinstance(embedding_params, dict) and "feature_scale" in embedding_params:
        return float(embedding_params["feature_scale"])
    return 1.0


def _build_pca_crowding_model(
    pos: dict[int, np.ndarray],
    isotropy_epsilon: float,
    seed: int | None,
) -> dict[str, Any]:
    """Build a fixed PCA crowding model from the original embedding."""

    if not pos:
        raise ValueError("Cannot build a crowding model from an empty embedding.")

    nodes = sorted(pos)
    stacked = np.stack([np.asarray(pos[node], dtype=float) for node in nodes], axis=0)
    dimension = int(stacked.shape[1])
    rng = random.Random(seed)

    centroid = np.mean(stacked, axis=0)
    centered = stacked - centroid
    covariance = (centered.T @ centered) / max(1, stacked.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    isotropic_fallback = False
    if dimension == 1:
        principal_axis = np.array([1.0], dtype=float)
    else:
        denom = max(float(eigenvalues[1]), 1e-12) if eigenvalues.shape[0] > 1 else 1e-12
        nearly_isotropic = float(eigenvalues[0]) <= 1e-12 or (
            eigenvalues.shape[0] > 1 and float(eigenvalues[0]) / denom <= 1.0 + isotropy_epsilon
        )
        if nearly_isotropic:
            isotropic_fallback = True
            principal_axis = random_unit_vector_nd(dimension, rng)
        else:
            principal_axis = np.asarray(eigenvectors[:, 0], dtype=float)
    principal_axis = normalize_nd(principal_axis)
    return {
        "nodes": nodes,
        "dimension": dimension,
        "centroid": centroid,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "principal_axis": principal_axis,
        "isotropic_fallback": isotropic_fallback,
    }


def _apply_pca_crowding_model(
    pos: dict[int, np.ndarray],
    crowding_model: dict[str, Any],
    gamma: float,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Apply a fixed PCA crowding model to node positions."""

    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must lie in (0, 1], got {gamma}.")

    nodes = list(crowding_model["nodes"])
    dimension = int(crowding_model["dimension"])
    centroid = np.asarray(crowding_model["centroid"], dtype=float)
    principal_axis = np.asarray(crowding_model["principal_axis"], dtype=float)
    stacked = np.stack([np.asarray(pos[node], dtype=float) for node in nodes], axis=0)
    centered = stacked - centroid
    projection = np.outer(principal_axis, principal_axis)
    operator = gamma * np.eye(dimension, dtype=float) + (1.0 - gamma) * projection
    transformed = centroid + centered @ operator.T
    crowded_pos = {node: transformed[idx].copy() for idx, node in enumerate(nodes)}
    return crowded_pos, operator


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return vector / norm


def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def _perpendicular(vector: np.ndarray) -> np.ndarray:
    return np.array([-vector[1], vector[0]], dtype=float)


def _random_unit_vector(rng: random.Random) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * math.pi)
    return np.array([math.cos(angle), math.sin(angle)], dtype=float)


def _order_branch_children(
    children: list[int],
    parent: int,
    branch_depth_map: dict[int, int],
) -> list[int]:
    parent_depth = branch_depth_map.get(parent)
    continuation = [
        child for child in children if parent_depth is not None and branch_depth_map.get(child) == parent_depth
    ]
    other_children = [child for child in children if child not in continuation]
    continuation.sort()
    other_children.sort()
    return continuation + other_children


def _best_branch_direction(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    parent: int,
    child: int,
    base_direction: np.ndarray,
    edge_length: float,
    max_local_placement_attempts: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    best_direction = _normalize(base_direction)
    best_position = pos[parent] + best_direction * edge_length
    best_clearance = candidate_branch_clearance(
        G=G,
        existing_pos=pos,
        parent=parent,
        child=child,
        child_position=best_position,
    )

    for attempt_idx in range(max_local_placement_attempts):
        angle_budget = _interpolate_attempt_span(
            attempt_idx=attempt_idx,
            max_attempts=max_local_placement_attempts,
            min_span=0.35,
            max_span=1.35,
        )
        direction = _rotate(base_direction, rng.uniform(-angle_budget, angle_budget))
        position = pos[parent] + _normalize(direction) * edge_length
        clearance = candidate_branch_clearance(
            G=G,
            existing_pos=pos,
            parent=parent,
            child=child,
            child_position=position,
        )
        if clearance > best_clearance:
            best_clearance = clearance
            best_direction = _normalize(direction)
            best_position = position

    return best_direction, best_position


def _best_branch_direction_3d(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    parent: int,
    child: int,
    base_direction: np.ndarray,
    edge_length: float,
    max_local_placement_attempts: int,
    rng: random.Random,
) -> np.ndarray:
    best_direction = _normalize_3d(base_direction)
    best_position = pos[parent] + best_direction * edge_length
    best_clearance = candidate_branch_clearance_3d(
        G=G,
        existing_pos=pos,
        parent=parent,
        child=child,
        child_position=best_position,
    )

    for attempt_idx in range(max_local_placement_attempts):
        angle_budget = _interpolate_attempt_span(
            attempt_idx=attempt_idx,
            max_attempts=max_local_placement_attempts,
            min_span=0.25,
            max_span=1.15,
        )
        direction = _sample_branch_direction_3d(
            base_direction=base_direction,
            angle_budget=angle_budget,
            rng=rng,
        )
        position = pos[parent] + _normalize_3d(direction) * edge_length
        clearance = candidate_branch_clearance_3d(
            G=G,
            existing_pos=pos,
            parent=parent,
            child=child,
            child_position=position,
        )
        if clearance > best_clearance:
            best_clearance = clearance
            best_direction = _normalize_3d(direction)
            best_position = position

    return best_position


def _sample_branch_direction_3d(
    base_direction: np.ndarray,
    angle_budget: float,
    rng: random.Random,
) -> np.ndarray:
    direction = _normalize_3d(base_direction)
    perturbation = random_unit_vector_3d(rng)
    candidate = direction + math.tan(rng.uniform(-angle_budget, angle_budget)) * perturbation
    if np.linalg.norm(candidate) < 1e-10:
        candidate = direction
    return _normalize_3d(candidate)


def _normalize_3d(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vector / norm


def _interpolate_attempt_span(
    attempt_idx: int,
    max_attempts: int,
    min_span: float,
    max_span: float,
) -> float:
    if max_attempts <= 1:
        return max_span
    progress = attempt_idx / max(1, max_attempts - 1)
    return min_span + (max_span - min_span) * progress
