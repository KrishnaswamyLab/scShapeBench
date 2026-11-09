"""Batch generator for the graph-to-point-cloud benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import networkx as nx
import numpy as np

from .cycle_backbone import (
    add_random_branches,
    apply_pca_crowding_by_relative_budget,
    embed_graph_constructively,
    generate_cycle_backbone,
    sample_noisy_point_cloud_from_embedding,
)

# ---------------------------------------------------------------------------
# User-tunable generation parameters
# ---------------------------------------------------------------------------

DEFAULT_SEED = 20260420
DEFAULT_DENSITY_JITTER_RATIO_RANGE = (0.0, 0.15)
DEFAULT_EPSILON_SEP = 0.02
DEFAULT_MAX_REJECTIONS_PER_SAMPLE = 100

THICKNESS_BUDGET_RANGE = (0.45, 1.00)
UNBOUNDED_THICKNESS_MAX_RADIUS_RATIO = 1.0
MIN_REALIZED_EDGE_TUBE_RADIUS_RATIO = 0.02
CYCLE_HOLE_RADIUS_FRACTION = 0.20

BASE_EMBEDDING_SEPARATION_RATIO = 6.0
EDGE_LENGTH_JITTER_RATIO_RANGE = (0.0, 0.12)
BRANCH_ANGLE_SPREAD_RANGE = (0.5, 1.2)
MAX_LOCAL_PLACEMENT_ATTEMPTS = 20
MAX_GLOBAL_RESAMPLING_ATTEMPTS = 5

CROWDING_ISOTROPY_EPSILON = 0.05
CROWDING_MIN_GAMMA = 0.01
CROWDING_SEARCH_TOLERANCE = 0.005
CROWDING_MAX_SEARCH_ITERS = 24

NOISE_CLEARANCE_SAFETY_FACTOR = 6.0
COMPONENT_NOISE_GAP_SIGMA_MULTIPLIER = 6.0
THICKNESS_NOISE_MARGIN_SIGMA_MULTIPLIER = 4.0

DIFFICULTY_PRESETS: dict[str, dict[str, tuple[float, float]]] = {
    "easy": {
        "noise_ratio": (0.02, 0.06),
        "crowding_r": (0.00, 0.20),
        "sampling_density_per_feature": (16.0, 25.0),
        "feature_thickness_ratio": THICKNESS_BUDGET_RANGE,
    },
    "medium": {
        "noise_ratio": (0.06, 0.13),
        "crowding_r": (0.20, 0.50),
        "sampling_density_per_feature": (8.0, 16.0),
        "feature_thickness_ratio": THICKNESS_BUDGET_RANGE,
    },
    "hard": {
        "noise_ratio": (0.13, 0.20),
        "crowding_r": (0.50, 0.60),
        "sampling_density_per_feature": (3.5, 8.0),
        "feature_thickness_ratio": THICKNESS_BUDGET_RANGE,
    },
    "full": {
        "noise_ratio": (0.02, 0.20),
        "crowding_r": (0.00, 0.60),
        "sampling_density_per_feature": (3.5, 25.0),
        "feature_thickness_ratio": THICKNESS_BUDGET_RANGE,
    },
}

DEFAULT_TOPOLOGY_CONFIG: dict[str, Any] = {
    "num_cycles_range": (0, 6),
    "cycle_len_range": (3, 8),
    "sep_range": (-1, 3),
    "branch_attach_count_range": (0, 4),
    "children_per_branch_point_range": (1, 3),
    "branch_length_range": (1, 4),
    "max_branch_depth_range": (1, 3),
    "promotion_prob_range": (0.2, 0.6),
    "max_degree": 5,
    "max_total_nodes": 80,
    "max_total_edges": 88,
}

REQUESTED_DIMENSION_PROBABILITIES: tuple[tuple[int, float], ...] = (
    (2, 0.50),
    (3, 0.35),
    (4, 0.10),
    (5, 0.05),
)

TUBE_SAMPLING_MODE_PROBABILITIES: tuple[tuple[str, float], ...] = (
    ("solid", 1.00),
)

DISCONNECTED_COMPONENT_CONFIG: dict[str, Any] = {
    "p_add_component": 0.25,
    "component_add_decay": 0.55,
    "max_components": 4,
    "component_gap_ratio_range": (4.0, 8.0),
    "min_points_per_component": 20,
}

TOPOLOGY_CLASS_PROBABILITIES: tuple[tuple[str, float], ...] = (
    ("singleton", 0.03),
    ("single_edge", 0.05),
    ("tree", 0.17),
    ("single_cycle", 0.20),
    ("multi_cycle", 0.25),
    ("branchy_cycle", 0.30),
)

NOISY_FEATURE_CONFUSION_THRESHOLDS: dict[str, dict[str, float]] = {
    "easy": {
        "nonincident_feature_confusion_rate": 0.02,
        "cross_component_confusion_rate": 0.0,
    },
    "medium": {
        "nonincident_feature_confusion_rate": 0.06,
        "cross_component_confusion_rate": 0.01,
    },
    "hard": {
        "nonincident_feature_confusion_rate": 0.12,
        "cross_component_confusion_rate": 0.03,
    },
    "full": {
        "nonincident_feature_confusion_rate": 0.12,
        "cross_component_confusion_rate": 0.03,
    },
}

METADATA_EXCLUDE_KEYS = {
    "clean_points",
    "noisy_points",
    "point_centerline_points",
    "point_tube_offsets",
    "point_noise_vectors",
    "point_source_type",
    "point_source_edge",
    "point_source_node",
    "point_source_t",
    "point_branch_depth",
    "point_branch_root",
    "point_cycle_id",
    "node_positions",
    "node_positions_2d",
    "node_positions_3d",
    "node_positions_pre_crowding",
    "node_positions_2d_pre_crowding",
    "node_positions_3d_pre_crowding",
}


def generate_benchmark_batch(
    output_dir: str | Path,
    num_samples: int,
    difficulty: str = "full",
    seed: int = DEFAULT_SEED,
    include_node_points: bool = True,
    min_points_per_edge: int = 2,
    density_jitter_ratio_range: tuple[float, float] = DEFAULT_DENSITY_JITTER_RATIO_RANGE,
    endpoint_sampling_mode: str = "include",
    epsilon_sep: float = DEFAULT_EPSILON_SEP,
    max_rejections_per_sample: int = DEFAULT_MAX_REJECTIONS_PER_SAMPLE,
) -> Path:
    """Generate a batch benchmark dataset and write it to disk."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(
            f"Unsupported difficulty {difficulty!r}. Choose from {sorted(DIFFICULTY_PRESETS)}."
        )
    if max_rejections_per_sample < 1:
        raise ValueError("max_rejections_per_sample must be positive.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    samples_dir = output_path / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    manifest_entries: list[dict[str, Any]] = []
    total_rejections = 0

    print(
        f"Generating {num_samples} samples to {output_path} "
        f"(difficulty={difficulty}, seed={seed})",
        flush=True,
    )

    for sample_idx in range(num_samples):
        sample_payload, sample_rejections = _generate_one_sample(
            sample_idx=sample_idx,
            difficulty=difficulty,
            rng=rng,
            include_node_points=include_node_points,
            min_points_per_edge=min_points_per_edge,
            density_jitter_ratio_range=density_jitter_ratio_range,
            endpoint_sampling_mode=endpoint_sampling_mode,
            epsilon_sep=epsilon_sep,
            max_rejections=max_rejections_per_sample,
        )
        total_rejections += sample_rejections

        sample_stem = f"sample_{sample_idx:05d}"
        npz_path = samples_dir / f"{sample_stem}.npz"
        json_path = samples_dir / f"{sample_stem}.json"

        np.savez_compressed(
            npz_path,
            nodes=np.asarray(sample_payload["nodes"], dtype=int),
            edges=np.asarray(sample_payload["edges"], dtype=int).reshape(-1, 2),
            positions=np.asarray(sample_payload["positions"], dtype=float),
            clean_points=np.asarray(sample_payload["clean_points"], dtype=float),
            noisy_points=np.asarray(sample_payload["noisy_points"], dtype=float),
        )
        json_path.write_text(
            json.dumps(_to_jsonable(sample_payload["sample_metadata"]), indent=2) + "\n",
            encoding="utf-8",
        )

        manifest_entries.append(
            {
                "sample_id": sample_payload["sample_id"],
                "difficulty": difficulty,
                "difficulty_coordinate": sample_payload["difficulty_coordinate"],
                "realized_geometry": sample_payload["realized_geometry"],
                "graph_summary": sample_payload["graph_summary"],
                "embedding_dimension": sample_payload["embedding_dimension"],
                "npz_path": str(npz_path.relative_to(output_path)),
                "json_path": str(json_path.relative_to(output_path)),
                "rejections_before_accept": sample_rejections,
            }
        )

        graph_summary = sample_payload["graph_summary"]
        difficulty_coordinate = sample_payload["difficulty_coordinate"]
        print(
            (
                f"[{sample_idx + 1}/{num_samples}] accepted sample_{sample_idx:05d} "
                f"after {sample_rejections} rejections | "
                f"nodes={graph_summary['num_nodes']} "
                f"edges={graph_summary['num_edges']} "
                f"points={graph_summary['num_clean_points']} "
                f"dim={sample_payload['embedding_dimension']} | "
                f"components={graph_summary['num_components']} "
                f"noise={difficulty_coordinate['noise_ratio']:.3f} "
                f"crowding_r={difficulty_coordinate['crowding_r']:.3f} "
                f"density={difficulty_coordinate['sampling_density_per_feature']:.3f} "
                f"thickness_budget={difficulty_coordinate['feature_thickness_ratio']:.3f}"
            ),
            flush=True,
        )

    config = {
        "num_samples": num_samples,
        "difficulty": difficulty,
        "difficulty_preset": DIFFICULTY_PRESETS[difficulty],
        "feature_scale_definition": "pre-crowding median edge length",
        "difficulty_coordinate_order": [
            "noise_ratio",
            "crowding_r",
            "sampling_density_per_feature",
            "feature_thickness_ratio",
            "num_components",
            "requested_embedding_dimension",
        ],
        "generation_policy": "sample_requested_dimension_then_constructive_embed_else_reject",
        "crowding_policy": "normalized_relative_budget_r",
        "crowding_epsilon_sep": epsilon_sep,
        "requested_dimension_probabilities": REQUESTED_DIMENSION_PROBABILITIES,
        "tube_sampling_mode_probabilities": TUBE_SAMPLING_MODE_PROBABILITIES,
        "disconnected_component_config": DISCONNECTED_COMPONENT_CONFIG,
        "noisy_feature_confusion_thresholds": NOISY_FEATURE_CONFUSION_THRESHOLDS,
        "include_node_points": include_node_points,
        "min_points_per_edge": min_points_per_edge,
        "density_jitter_ratio_range": density_jitter_ratio_range,
        "endpoint_sampling_mode": endpoint_sampling_mode,
        "seed": seed,
        "topology_config": DEFAULT_TOPOLOGY_CONFIG,
        "total_rejections_before_acceptance": total_rejections,
    }
    (output_path / "dataset_config.json").write_text(
        json.dumps(_to_jsonable(config), indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_path / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for entry in manifest_entries:
            handle.write(json.dumps(_to_jsonable(entry)) + "\n")

    print(
        f"Finished {num_samples} samples with {total_rejections} total rejections. "
        f"Wrote dataset to {output_path}",
        flush=True,
    )

    return output_path


def _generate_one_sample(
    sample_idx: int,
    difficulty: str,
    rng: random.Random,
    include_node_points: bool,
    min_points_per_edge: int,
    density_jitter_ratio_range: tuple[float, float],
    endpoint_sampling_mode: str,
    epsilon_sep: float,
    max_rejections: int,
    difficulty_coordinate_override: tuple[float, ...] | None = None,
) -> tuple[dict[str, Any], int]:
    """Generate one accepted sample and return it with its rejection count."""

    for rejection_count in range(max_rejections):
        global_params = _sample_global_generation_params(
            rng=rng,
            difficulty=difficulty,
            difficulty_coordinate_override=difficulty_coordinate_override,
        )
        requested_noise_ratio = global_params["noise_ratio"]
        requested_crowding_r = global_params["crowding_r"]
        density = global_params["sampling_density_per_feature"]
        feature_thickness_budget = global_params["feature_thickness_ratio"]
        feature_scale = 1.0

        num_components = _sample_num_components(rng)
        component_gap_ratio = rng.uniform(
            *DISCONNECTED_COMPONENT_CONFIG["component_gap_ratio_range"]
        )
        point_cloud_seed = rng.randint(0, 10**9)

        try:
            component_payloads = [
                _generate_connected_component(
                    rng=rng,
                    density=density,
                    requested_embedding_dimension=global_params["requested_embedding_dimension"],
                )
                for _ in range(num_components)
            ]
        except ValueError:
            continue

        graph, base_pos, component_metadata = _assemble_disconnected_components(
            component_payloads=component_payloads,
            component_gap_ratio=component_gap_ratio,
            feature_scale=feature_scale,
            sigma=0.0,
        )
        if not _within_topology_size_caps(graph):
            continue

        metadata = _build_union_metadata(
            graph=graph,
            crowded_pos=base_pos,
            component_metadata=component_metadata,
            global_params=global_params,
            feature_scale=feature_scale,
            epsilon_sep=epsilon_sep,
            min_required_clearance=epsilon_sep * feature_scale,
            component_gap_ratio=component_gap_ratio,
        )

        base_stats = _union_geometry_stats(graph, base_pos)
        base_sep = float(base_stats["realized_post_crowding_min_structural_distance"])
        if np.isfinite(base_sep):
            max_radius = max(0.0, 0.5 * (base_sep - epsilon_sep * feature_scale))
        else:
            max_radius = float("inf")
        cycle_hole_radius_cap = _cycle_hole_radius_cap(
            graph=graph,
            pos=base_pos,
            fraction=CYCLE_HOLE_RADIUS_FRACTION,
        )
        if np.isfinite(cycle_hole_radius_cap):
            max_radius = min(max_radius, cycle_hole_radius_cap)
        effective_max_radius = (
            max_radius
            if np.isfinite(max_radius)
            else UNBOUNDED_THICKNESS_MAX_RADIUS_RATIO * feature_scale
        )
        effective_max_radius = max(0.0, float(effective_max_radius))
        feature_thickness_radius = float(feature_thickness_budget) * effective_max_radius
        metadata["feature_thickness_budget"] = float(feature_thickness_budget)
        metadata["feature_thickness_radius"] = float(feature_thickness_radius)
        metadata["effective_max_tube_radius_for_sampling"] = float(effective_max_radius)
        metadata["cycle_hole_radius_cap"] = float(cycle_hole_radius_cap)
        metadata["cycle_hole_radius_fraction"] = float(CYCLE_HOLE_RADIUS_FRACTION)
        if (
            graph.number_of_edges() > 0
            and feature_thickness_radius
            < MIN_REALIZED_EDGE_TUBE_RADIUS_RATIO * float(feature_scale)
        ):
            continue

        crowding_clearance = (
            2.0 * float(feature_thickness_radius)
            + float(epsilon_sep) * float(feature_scale)
        )
        try:
            crowded_pos, metadata = apply_pca_crowding_by_relative_budget(
                G=graph,
                pos=base_pos,
                metadata=metadata,
                r=float(requested_crowding_r),
                isotropy_epsilon=CROWDING_ISOTROPY_EPSILON,
                epsilon_sep=crowding_clearance,
                min_gamma=CROWDING_MIN_GAMMA,
                search_tolerance=CROWDING_SEARCH_TOLERANCE,
                max_search_iters=CROWDING_MAX_SEARCH_ITERS,
                seed=rng.randint(0, 10**9),
            )
        except ValueError:
            continue

        metadata["node_positions"] = _serialize_positions(crowded_pos)
        metadata.update(_union_geometry_stats(graph, crowded_pos))
        metadata["minimum_realized_component_gap"] = _minimum_component_gap(component_metadata)
        post_sep = float(metadata["realized_post_crowding_min_structural_distance"])
        if np.isfinite(post_sep):
            sigma_max = max(
                0.0,
                (post_sep - 2.0 * float(feature_thickness_radius) - float(epsilon_sep) * feature_scale)
                / NOISE_CLEARANCE_SAFETY_FACTOR,
            )
        else:
            sigma_max = float(requested_noise_ratio) * feature_scale
        noise_ratio = min(float(requested_noise_ratio), float(sigma_max) / feature_scale)
        sigma = noise_ratio * feature_scale

        metadata["embedding_params"]["noise_ratio"] = float(noise_ratio)
        metadata["requested_noise_ratio"] = float(requested_noise_ratio)
        metadata["realized_noise_ratio"] = float(noise_ratio)
        metadata["max_admissible_noise_ratio_after_crowding"] = float(sigma_max / feature_scale)
        metadata["requested_crowding_r"] = float(requested_crowding_r)
        metadata["realized_crowding_r"] = float(metadata["crowding_r"])
        metadata["staged_generation_order"] = (
            "topology -> max-separation embedding -> admissible thickness "
            "-> thickness-constrained crowding -> clearance-capped noise -> sampling"
        )
        metadata.update(
            _thickness_clearance_metadata(
                metadata=metadata,
                feature_thickness_radius=feature_thickness_radius,
                feature_scale=feature_scale,
                noise_ratio=float(noise_ratio),
                epsilon_sep=epsilon_sep,
            )
        )
        if not np.isfinite(max_radius):
            metadata["realized_tube_radius_budget_fraction"] = float(feature_thickness_budget)
        component_gap_required = (
            2.0 * float(feature_thickness_radius)
            + COMPONENT_NOISE_GAP_SIGMA_MULTIPLIER * sigma
            + float(epsilon_sep) * float(feature_scale)
        )
        if metadata["minimum_realized_component_gap"] < component_gap_required:
            continue
        if not metadata["tube_no_overlap_satisfied"]:
            continue

        clean_points, noisy_points, metadata = sample_noisy_point_cloud_from_embedding(
            G=graph,
            pos=crowded_pos,
            metadata=metadata,
            include_node_points=include_node_points,
            min_points_per_edge=min_points_per_edge,
            density_jitter_ratio_range=density_jitter_ratio_range,
            endpoint_sampling_mode=endpoint_sampling_mode,
            feature_thickness_ratio=feature_thickness_radius / float(feature_scale),
            tube_sampling_mode=global_params["tube_sampling_mode"],
            seed=point_cloud_seed,
        )
        _annotate_point_components(metadata, component_metadata)
        confusion = _compute_noisy_feature_confusion(
            G=graph,
            pos=crowded_pos,
            clean_points=clean_points,
            noisy_points=noisy_points,
            metadata=metadata,
            difficulty=difficulty,
        )
        metadata.update(confusion)

        component_point_counts = [
            len(ids) for ids in metadata.get("component_point_ids", [])
        ]
        if (
            component_point_counts
            and min(component_point_counts) < DISCONNECTED_COMPONENT_CONFIG["min_points_per_component"]
        ):
            continue

        node_ids = sorted(graph.nodes())
        positions = np.stack([crowded_pos[node] for node in node_ids], axis=0)
        edges = np.asarray(
            sorted((min(u, v), max(u, v)) for u, v in graph.edges()),
            dtype=int,
        ).reshape(-1, 2)
        difficulty_coordinate = {
            "noise_ratio": float(requested_noise_ratio),
            "crowding_r": float(requested_crowding_r),
            "sampling_density_per_feature": float(density),
            "feature_thickness_ratio": float(feature_thickness_budget),
            "num_components": int(num_components),
            "requested_embedding_dimension": int(global_params["requested_embedding_dimension"]),
        }
        realized_geometry = {
            "realized_separation_to_feature_scale": float(
                metadata["realized_post_crowding_separation_to_feature_scale"]
            ),
            "realized_absolute_hardness": float(
                metadata["realized_post_crowding_absolute_hardness"]
            ),
            "crowding_gamma": float(metadata["crowding_gamma"]),
            "crowding_gamma_min": float(metadata["crowding_gamma_min"]),
            "feature_thickness_radius": float(feature_thickness_radius),
            "feature_thickness_budget": float(feature_thickness_budget),
            "effective_max_tube_radius_for_sampling": float(effective_max_radius),
            "max_admissible_tube_radius_before_crowding": float(effective_max_radius),
            "cycle_hole_radius_cap": float(cycle_hole_radius_cap),
            "cycle_hole_radius_fraction": float(CYCLE_HOLE_RADIUS_FRACTION),
            "requested_noise_ratio": float(requested_noise_ratio),
            "realized_noise_ratio": float(noise_ratio),
            "max_admissible_noise_ratio_after_crowding": float(sigma_max / feature_scale),
            "requested_crowding_r": float(requested_crowding_r),
            "realized_crowding_r": float(metadata["crowding_r"]),
            "thickness_aware_min_required_clearance": float(crowding_clearance),
            "thickness_clearance_noise_margin": float(
                THICKNESS_NOISE_MARGIN_SIGMA_MULTIPLIER * sigma
            ),
            "minimum_realized_component_gap": float(metadata["minimum_realized_component_gap"]),
        }
        graph_summary = {
            "num_nodes": int(graph.number_of_nodes()),
            "num_edges": int(graph.number_of_edges()),
            "num_components": int(nx.number_connected_components(graph)),
            "num_clean_points": int(clean_points.shape[0]),
            "topology_classes": list(metadata.get("component_topology_classes", [])),
        }
        sample_metadata = _strip_large_metadata(metadata)
        sample_metadata["sample_id"] = sample_idx
        sample_metadata["difficulty"] = difficulty
        sample_metadata["difficulty_coordinate"] = difficulty_coordinate
        sample_metadata["realized_geometry"] = realized_geometry
        sample_metadata["graph_summary"] = graph_summary
        sample_metadata["seed_bundle"] = {
            "point_cloud_seed": point_cloud_seed,
            "component_seed_bundles": [
                item["seed_bundle"] for item in component_metadata
            ],
        }

        return {
            "sample_id": sample_idx,
            "nodes": node_ids,
            "edges": edges.tolist(),
            "positions": positions,
            "clean_points": clean_points,
            "noisy_points": noisy_points,
            "difficulty_coordinate": difficulty_coordinate,
            "realized_geometry": realized_geometry,
            "graph_summary": graph_summary,
            "embedding_dimension": int(metadata["embedding_dimension"]),
            "sample_metadata": sample_metadata,
        }, rejection_count

    raise RuntimeError(
        f"Failed to generate sample {sample_idx} after {max_rejections} rejected attempts."
    )


def _generate_connected_component(
    rng: random.Random,
    density: float,
    requested_embedding_dimension: int,
) -> dict[str, Any]:
    topology_spec = _sample_topology_spec(rng)
    num_cycles = int(topology_spec["num_cycles"])
    branch_attach_count = int(topology_spec["branch_attach_count"])

    topology_seed = rng.randint(0, 10**9)
    branch_seed = rng.randint(0, 10**9)
    embedding_seed = rng.randint(0, 10**9)

    graph, metadata = generate_cycle_backbone(
        num_cycles=num_cycles,
        cycle_len_range=DEFAULT_TOPOLOGY_CONFIG["cycle_len_range"],
        sep_range=DEFAULT_TOPOLOGY_CONFIG["sep_range"],
        seed=topology_seed,
    )
    if not _within_topology_size_caps(graph):
        raise ValueError("Rejected component exceeding topology size caps.")

    graph, metadata = add_random_branches(
        G=graph,
        metadata=metadata,
        branch_attach_count_range=(branch_attach_count, branch_attach_count),
        children_per_branch_point_range=topology_spec["children_per_branch_point_range"],
        branch_length_range=topology_spec["branch_length_range"],
        max_branch_depth_range=topology_spec["max_branch_depth_range"],
        promotion_prob_range=topology_spec["promotion_prob_range"],
        max_degree=int(topology_spec["max_degree"]),
        max_total_nodes=DEFAULT_TOPOLOGY_CONFIG["max_total_nodes"],
        max_total_edges=DEFAULT_TOPOLOGY_CONFIG["max_total_edges"],
        seed=branch_seed,
    )

    pos, metadata = embed_graph_constructively(
        G=graph,
        metadata=metadata,
        feature_scale_range=(1.0, 1.0),
        separation_to_feature_ratio_range=(
            BASE_EMBEDDING_SEPARATION_RATIO,
            BASE_EMBEDDING_SEPARATION_RATIO,
        ),
        noise_to_feature_ratio_range=(0.0, 0.0),
        sampling_density_per_feature_range=(density, density),
        edge_length_jitter_ratio_range=EDGE_LENGTH_JITTER_RATIO_RANGE,
        branch_angle_spread_range=BRANCH_ANGLE_SPREAD_RANGE,
        max_local_placement_attempts=MAX_LOCAL_PLACEMENT_ATTEMPTS,
        max_global_resampling_attempts=MAX_GLOBAL_RESAMPLING_ATTEMPTS,
        embedding_dimensions=(int(requested_embedding_dimension),),
        max_embedding_dimension=int(requested_embedding_dimension),
        seed=embedding_seed,
    )
    if metadata["embedding_rejected"]:
        raise ValueError("Rejected failed component embedding.")

    metadata["requested_embedding_dimension"] = int(requested_embedding_dimension)
    metadata["topology_class"] = topology_spec["topology_class"]
    metadata["sampled_topology_spec"] = topology_spec
    metadata["tube_embedding_separation_ratio"] = float(BASE_EMBEDDING_SEPARATION_RATIO)
    return {
        "graph": graph,
        "positions": pos,
        "metadata": metadata,
        "seed_bundle": {
            "topology_seed": topology_seed,
            "branch_seed": branch_seed,
            "embedding_seed": embedding_seed,
            "topology_class": topology_spec["topology_class"],
        },
    }


def _sample_topology_spec(rng: random.Random) -> dict[str, Any]:
    topology_class = _weighted_choice(rng, TOPOLOGY_CLASS_PROBABILITIES)
    base = {
        "topology_class": topology_class,
        "children_per_branch_point_range": DEFAULT_TOPOLOGY_CONFIG[
            "children_per_branch_point_range"
        ],
        "branch_length_range": DEFAULT_TOPOLOGY_CONFIG["branch_length_range"],
        "max_branch_depth_range": DEFAULT_TOPOLOGY_CONFIG["max_branch_depth_range"],
        "promotion_prob_range": DEFAULT_TOPOLOGY_CONFIG["promotion_prob_range"],
        "max_degree": DEFAULT_TOPOLOGY_CONFIG["max_degree"],
    }
    if topology_class == "singleton":
        base.update(
            {
                "num_cycles": 0,
                "branch_attach_count": 0,
                "children_per_branch_point_range": (1, 1),
                "branch_length_range": (1, 1),
                "max_branch_depth_range": (1, 1),
                "promotion_prob_range": (0.0, 0.0),
                "max_degree": 1,
            }
        )
    elif topology_class == "single_edge":
        base.update(
            {
                "num_cycles": 0,
                "branch_attach_count": 1,
                "children_per_branch_point_range": (1, 1),
                "branch_length_range": (1, 1),
                "max_branch_depth_range": (1, 1),
                "promotion_prob_range": (0.0, 0.0),
                "max_degree": 2,
            }
        )
    elif topology_class == "tree":
        base.update(
            {
                "num_cycles": 0,
                "branch_attach_count": rng.randint(1, 3),
                "max_branch_depth_range": (1, 2),
                "max_degree": min(4, int(DEFAULT_TOPOLOGY_CONFIG["max_degree"])),
            }
        )
    elif topology_class == "single_cycle":
        base.update(
            {
                "num_cycles": 1,
                "branch_attach_count": rng.randint(0, 1),
                "max_branch_depth_range": (1, 1),
                "max_degree": min(4, int(DEFAULT_TOPOLOGY_CONFIG["max_degree"])),
            }
        )
    elif topology_class == "multi_cycle":
        base.update(
            {
                "num_cycles": rng.randint(2, 4),
                "branch_attach_count": rng.randint(0, 2),
                "max_branch_depth_range": (1, 2),
                "max_degree": min(4, int(DEFAULT_TOPOLOGY_CONFIG["max_degree"])),
            }
        )
    elif topology_class == "branchy_cycle":
        base.update(
            {
                "num_cycles": rng.randint(2, DEFAULT_TOPOLOGY_CONFIG["num_cycles_range"][1]),
                "branch_attach_count": rng.randint(2, DEFAULT_TOPOLOGY_CONFIG["branch_attach_count_range"][1]),
            }
        )
    else:
        raise ValueError(f"Unsupported topology class {topology_class!r}.")
    return base


def _assemble_disconnected_components(
    component_payloads: list[dict[str, Any]],
    component_gap_ratio: float,
    feature_scale: float,
    sigma: float,
) -> tuple[nx.Graph, dict[int, np.ndarray], list[dict[str, Any]]]:
    graph = nx.Graph()
    positions: dict[int, np.ndarray] = {}
    component_metadata: list[dict[str, Any]] = []
    next_node_id = 0
    cursor_x = 0.0
    previous_max_x: float | None = None
    gap = (
        component_gap_ratio * feature_scale
        + COMPONENT_NOISE_GAP_SIGMA_MULTIPLIER * sigma
    )

    for component_id, payload in enumerate(component_payloads):
        component_graph: nx.Graph = payload["graph"]
        component_pos = {
            int(node): np.asarray(coords, dtype=float)
            for node, coords in payload["positions"].items()
        }
        node_order = sorted(component_graph.nodes())
        mapping = {node: next_node_id + idx for idx, node in enumerate(node_order)}
        relabeled = nx.relabel_nodes(component_graph, mapping, copy=True)

        coords = np.vstack([component_pos[node] for node in node_order])
        min_x = float(np.min(coords[:, 0]))
        max_x = float(np.max(coords[:, 0]))
        if previous_max_x is None:
            shift_x = -min_x
        else:
            shift_x = cursor_x + gap - min_x
        translation = np.zeros(coords.shape[1], dtype=float)
        translation[0] = shift_x

        graph.add_nodes_from(relabeled.nodes())
        graph.add_edges_from(relabeled.edges())
        global_nodes = []
        for old_node in node_order:
            new_node = mapping[old_node]
            positions[new_node] = component_pos[old_node] + translation
            global_nodes.append(int(new_node))

        shifted_min_x = min_x + shift_x
        shifted_max_x = max_x + shift_x
        if previous_max_x is not None:
            cursor_x = shifted_max_x
        else:
            cursor_x = shifted_max_x
        previous_max_x = shifted_max_x

        component_metadata.append(
            {
                "component_id": int(component_id),
                "node_ids": global_nodes,
                "x_interval": [float(shifted_min_x), float(shifted_max_x)],
                "source_metadata": payload["metadata"],
                "seed_bundle": payload["seed_bundle"],
            }
        )
        next_node_id += len(node_order)

    return graph, positions, component_metadata


def _build_union_metadata(
    graph: nx.Graph,
    crowded_pos: dict[int, np.ndarray],
    component_metadata: list[dict[str, Any]],
    global_params: dict[str, Any],
    feature_scale: float,
    epsilon_sep: float,
    min_required_clearance: float,
    component_gap_ratio: float,
) -> dict[str, Any]:
    first_source = component_metadata[0]["source_metadata"]
    metadata: dict[str, Any] = {
        "embedding_params": {
            "feature_scale": float(feature_scale),
            "sampling_density_per_feature": float(global_params["sampling_density_per_feature"]),
            "noise_ratio": float(global_params["noise_ratio"]),
        },
        "embedding_dimension": int(global_params["requested_embedding_dimension"]),
        "requested_embedding_dimension": int(global_params["requested_embedding_dimension"]),
        "feature_thickness_ratio": float(global_params["feature_thickness_ratio"]),
        "feature_thickness_budget": float(global_params["feature_thickness_ratio"]),
        "tube_sampling_mode": global_params["tube_sampling_mode"],
        "num_components": len(component_metadata),
        "component_gap_ratio": float(component_gap_ratio),
        "component_node_ids": [item["node_ids"] for item in component_metadata],
        "component_x_intervals": [item["x_interval"] for item in component_metadata],
        "component_topology_classes": [
            item["source_metadata"].get("topology_class", "unknown")
            for item in component_metadata
        ],
        "crowding_r": float(global_params["crowding_r"]),
        "crowding_gamma": float(first_source.get("crowding_gamma", np.nan)),
        "crowding_gamma_min": float(first_source.get("crowding_gamma_min", np.nan)),
        "crowding_admissibility_epsilon_sep": float(epsilon_sep),
        "thickness_aware_min_required_clearance": float(min_required_clearance),
        "thickness_crowding_order": "choose_tube_radius_then_apply_thickness_aware_crowding",
        "node_positions": _serialize_positions(crowded_pos),
    }
    if len(component_metadata) == 1:
        metadata.update(
            {
                key: value
                for key, value in first_source.items()
                if key not in {"node_positions", "node_positions_2d", "node_positions_3d"}
            }
        )
        metadata["num_components"] = 1
        metadata["component_node_ids"] = [component_metadata[0]["node_ids"]]

    stats = _union_geometry_stats(graph, crowded_pos)
    metadata.update(stats)
    metadata["minimum_realized_component_gap"] = _minimum_component_gap(component_metadata)
    return metadata


def _annotate_point_components(
    metadata: dict[str, Any],
    component_metadata: list[dict[str, Any]],
) -> None:
    node_to_component = {}
    for item in component_metadata:
        for node in item["node_ids"]:
            node_to_component[int(node)] = int(item["component_id"])

    point_component_ids = []
    component_point_ids = [[] for _ in component_metadata]
    source_nodes = metadata.get("point_source_node", [])
    source_edges = metadata.get("point_source_edge", [])
    for point_id, (source_node, source_edge) in enumerate(zip(source_nodes, source_edges)):
        if source_node is not None:
            component_id = node_to_component[int(source_node)]
        elif source_edge is not None:
            component_id = node_to_component[int(source_edge[0])]
        else:
            component_id = -1
        point_component_ids.append(component_id)
        if component_id >= 0:
            component_point_ids[component_id].append(point_id)

    metadata["point_component_id"] = point_component_ids
    metadata["component_point_ids"] = component_point_ids


def _compute_noisy_feature_confusion(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    clean_points: np.ndarray,
    noisy_points: np.ndarray,
    metadata: dict[str, Any],
    difficulty: str,
) -> dict[str, Any]:
    """Measure noisy Voronoi-cell changes relative to graph centerline features."""

    features = _graph_feature_records(G=G, pos=pos)
    if not features or clean_points.shape[0] == 0:
        thresholds = NOISY_FEATURE_CONFUSION_THRESHOLDS[difficulty]
        return {
            "nearest_feature_changed_rate": 0.0,
            "nonincident_feature_confusion_rate": 0.0,
            "cross_component_confusion_rate": 0.0,
            "noise_overlap_quality_pass": True,
            "noise_overlap_quality_thresholds": thresholds,
            "num_points_evaluated_for_noise_confusion": int(clean_points.shape[0]),
        }

    source_nodes = metadata.get("point_source_node", [])
    source_edges = metadata.get("point_source_edge", [])
    source_components = metadata.get("point_component_id", [])
    thresholds = NOISY_FEATURE_CONFUSION_THRESHOLDS[difficulty]

    changed = 0
    nonincident = 0
    cross_component = 0
    evaluated = int(clean_points.shape[0])
    for idx, (clean_point, noisy_point) in enumerate(zip(clean_points, noisy_points)):
        clean_feature = _nearest_graph_feature(clean_point, features)
        noisy_feature = _nearest_graph_feature(noisy_point, features)
        if clean_feature["id"] != noisy_feature["id"]:
            changed += 1

        source_node = source_nodes[idx] if idx < len(source_nodes) else None
        source_edge = source_edges[idx] if idx < len(source_edges) else None
        if not _feature_is_source_equivalent(
            feature_id=noisy_feature["id"],
            source_node=source_node,
            source_edge=source_edge,
        ):
            nonincident += 1

        source_component = source_components[idx] if idx < len(source_components) else None
        if source_component is not None and source_component >= 0:
            if int(noisy_feature["component_id"]) != int(source_component):
                cross_component += 1

    changed_rate = changed / evaluated
    nonincident_rate = nonincident / evaluated
    cross_component_rate = cross_component / evaluated
    quality_pass = (
        nonincident_rate <= thresholds["nonincident_feature_confusion_rate"]
        and cross_component_rate <= thresholds["cross_component_confusion_rate"]
    )
    return {
        "nearest_feature_changed_rate": float(changed_rate),
        "nonincident_feature_confusion_rate": float(nonincident_rate),
        "cross_component_confusion_rate": float(cross_component_rate),
        "noise_overlap_quality_pass": bool(quality_pass),
        "noise_overlap_quality_thresholds": thresholds,
        "num_points_evaluated_for_noise_confusion": evaluated,
        "num_nearest_feature_changes": int(changed),
        "num_nonincident_feature_confusions": int(nonincident),
        "num_cross_component_confusions": int(cross_component),
    }


def _graph_feature_records(
    G: nx.Graph,
    pos: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    components = {
        node: component_id
        for component_id, nodes in enumerate(nx.connected_components(G))
        for node in nodes
    }
    records: list[dict[str, Any]] = []
    for node in sorted(G.nodes()):
        records.append(
            {
                "id": ("node", int(node)),
                "kind": "node",
                "nodes": (int(node),),
                "component_id": int(components[node]),
                "point": np.asarray(pos[node], dtype=float),
            }
        )
    for u, v in sorted((min(u, v), max(u, v)) for u, v in G.edges()):
        records.append(
            {
                "id": ("edge", int(u), int(v)),
                "kind": "edge",
                "nodes": (int(u), int(v)),
                "component_id": int(components[u]),
                "p0": np.asarray(pos[u], dtype=float),
                "p1": np.asarray(pos[v], dtype=float),
            }
        )
    return records


def _nearest_graph_feature(point: np.ndarray, features: list[dict[str, Any]]) -> dict[str, Any]:
    point = np.asarray(point, dtype=float)
    best_feature = features[0]
    best_distance = float("inf")
    for feature in features:
        if feature["kind"] == "node":
            distance = float(np.linalg.norm(point - feature["point"]))
        else:
            distance = _point_segment_distance(point, feature["p0"], feature["p1"])
        if distance < best_distance:
            best_distance = distance
            best_feature = feature
    return best_feature


def _point_segment_distance(point: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    segment = p1 - p0
    denom = float(np.dot(segment, segment))
    if denom <= np.finfo(float).eps:
        return float(np.linalg.norm(point - p0))
    t = float(np.clip(np.dot(point - p0, segment) / denom, 0.0, 1.0))
    projection = p0 + t * segment
    return float(np.linalg.norm(point - projection))


def _feature_is_source_equivalent(
    feature_id: tuple[Any, ...],
    source_node: Any,
    source_edge: Any,
) -> bool:
    feature_kind = feature_id[0]
    if source_node is not None:
        node = int(source_node)
        if feature_kind == "node":
            return int(feature_id[1]) == node
        if feature_kind == "edge":
            return node in {int(feature_id[1]), int(feature_id[2])}
        return False

    if source_edge is not None:
        u, v = sorted((int(source_edge[0]), int(source_edge[1])))
        if feature_kind == "edge":
            return (int(feature_id[1]), int(feature_id[2])) == (u, v)
        if feature_kind == "node":
            return int(feature_id[1]) in {u, v}
        return False

    return True


def _union_geometry_stats(
    graph: nx.Graph,
    pos: dict[int, np.ndarray],
) -> dict[str, float]:
    from .geometry_nd import validate_embedding_nd

    _, stats = validate_embedding_nd(
        G=graph,
        pos=pos,
        target_min_separation=0.0,
        tolerance=0.0,
    )
    edge_lengths = [
        float(np.linalg.norm(np.asarray(pos[u]) - np.asarray(pos[v])))
        for u, v in graph.edges()
    ]
    median_edge = float(np.median(edge_lengths)) if edge_lengths else 1.0
    finite_separations = [
        value for value in (
            float(stats["min_nonadjacent_node_distance"]),
            float(stats["min_nonincident_edge_distance"]),
            float(stats["min_nonincident_node_edge_distance"]),
        )
        if np.isfinite(value)
    ]
    realized_sep = min(finite_separations) if finite_separations else float("inf")
    return {
        "realized_post_crowding_min_nonadjacent_node_distance": float(
            stats["min_nonadjacent_node_distance"]
        ),
        "realized_post_crowding_min_nonincident_edge_distance": float(
            stats["min_nonincident_edge_distance"]
        ),
        "realized_post_crowding_min_nonincident_node_edge_distance": float(
            stats["min_nonincident_node_edge_distance"]
        ),
        "realized_post_crowding_min_structural_distance": float(realized_sep),
        "realized_post_crowding_separation_to_feature_scale": float(realized_sep),
        "realized_post_crowding_separation_to_post_median_edge": float(realized_sep / median_edge),
        "realized_post_crowding_absolute_hardness": float(1.0 / realized_sep)
        if realized_sep > 0.0 and np.isfinite(realized_sep)
        else float("inf"),
    }


def _cycle_hole_radius_cap(
    graph: nx.Graph,
    pos: dict[int, np.ndarray],
    fraction: float,
) -> float:
    """Return a tube-radius cap that keeps sampled cycle interiors visible."""

    cycle_caps = []
    for cycle_nodes in nx.cycle_basis(graph):
        if len(cycle_nodes) < 3:
            continue
        coords = [np.asarray(pos[node], dtype=float) for node in cycle_nodes]
        center = np.mean(np.stack(coords), axis=0)
        distances = []
        for idx, node in enumerate(cycle_nodes):
            next_node = cycle_nodes[(idx + 1) % len(cycle_nodes)]
            distances.append(
                _point_segment_distance(
                    center,
                    np.asarray(pos[node], dtype=float),
                    np.asarray(pos[next_node], dtype=float),
                )
            )
        finite_distances = [distance for distance in distances if np.isfinite(distance)]
        if finite_distances:
            cycle_caps.append(float(fraction) * min(finite_distances))
    if not cycle_caps:
        return float("inf")
    return max(0.0, min(cycle_caps))


def _thickness_clearance_metadata(
    metadata: dict[str, Any],
    feature_thickness_radius: float,
    feature_scale: float,
    noise_ratio: float,
    epsilon_sep: float,
) -> dict[str, float | bool]:
    """Summarize whether post-crowding centerlines can hold thickened tubes."""

    sigma = float(noise_ratio) * float(feature_scale)
    noise_margin = THICKNESS_NOISE_MARGIN_SIGMA_MULTIPLIER * sigma
    no_overlap_clearance = (
        2.0 * float(feature_thickness_radius)
        + noise_margin
        + float(epsilon_sep) * float(feature_scale)
    )
    realized = float(metadata.get("realized_post_crowding_min_structural_distance", float("inf")))
    if np.isfinite(realized):
        margin = realized - no_overlap_clearance
        max_radius = max(0.0, 0.5 * (realized - noise_margin - float(epsilon_sep) * float(feature_scale)))
    else:
        margin = float("inf")
        max_radius = float("inf")
    if np.isfinite(max_radius) and max_radius > 0.0:
        budget_fraction = float(feature_thickness_radius) / max_radius
    elif not np.isfinite(max_radius):
        budget_fraction = 0.0
    else:
        budget_fraction = 0.0 if float(feature_thickness_radius) <= 0.0 else float("inf")
    return {
        "tube_no_overlap_required_centerline_clearance": float(no_overlap_clearance),
        "tube_no_overlap_post_crowding_margin": float(margin),
        "tube_no_overlap_satisfied": bool(margin >= -1e-9),
        "tube_no_overlap_constraint_active": bool(np.isfinite(realized)),
        "max_admissible_tube_radius_after_crowding": float(max_radius),
        "realized_tube_radius_budget_fraction": float(budget_fraction),
    }


def _minimum_component_gap(component_metadata: list[dict[str, Any]]) -> float:
    if len(component_metadata) <= 1:
        return float("inf")
    gaps = []
    intervals = [item["x_interval"] for item in component_metadata]
    for left, right in zip(intervals, intervals[1:]):
        gaps.append(float(right[0]) - float(left[1]))
    return min(gaps) if gaps else float("inf")


def _serialize_positions(pos: dict[int, np.ndarray]) -> dict[str, list[float]]:
    return {str(node): np.asarray(coords, dtype=float).tolist() for node, coords in pos.items()}


def _within_topology_size_caps(graph) -> bool:
    max_total_nodes = DEFAULT_TOPOLOGY_CONFIG["max_total_nodes"]
    max_total_edges = DEFAULT_TOPOLOGY_CONFIG["max_total_edges"]
    return (
        graph.number_of_nodes() <= max_total_nodes
        and graph.number_of_edges() <= max_total_edges
    )


def _sample_difficulty_coordinate(
    rng: random.Random,
    difficulty: str,
) -> tuple[float, float, float, float]:
    preset = DIFFICULTY_PRESETS[difficulty]
    noise_ratio = rng.uniform(*preset["noise_ratio"])
    crowding_r = rng.uniform(*preset["crowding_r"])
    density = rng.uniform(*preset["sampling_density_per_feature"])
    thickness = rng.uniform(*preset["feature_thickness_ratio"])
    return noise_ratio, crowding_r, density, thickness


def _sample_global_generation_params(
    rng: random.Random,
    difficulty: str,
    difficulty_coordinate_override: tuple[float, ...] | None,
) -> dict[str, Any]:
    if difficulty_coordinate_override is None:
        noise_ratio, crowding_r, density, thickness = _sample_difficulty_coordinate(
            rng=rng,
            difficulty=difficulty,
        )
    else:
        if len(difficulty_coordinate_override) == 3:
            noise_ratio, crowding_r, density = difficulty_coordinate_override
            thickness = rng.uniform(*DIFFICULTY_PRESETS[difficulty]["feature_thickness_ratio"])
        elif len(difficulty_coordinate_override) == 4:
            noise_ratio, crowding_r, density, thickness = difficulty_coordinate_override
        else:
            raise ValueError("difficulty_coordinate_override must have length 3 or 4.")

    return {
        "noise_ratio": float(noise_ratio),
        "crowding_r": float(crowding_r),
        "sampling_density_per_feature": float(density),
        "feature_thickness_ratio": float(thickness),
        "tube_sampling_mode": _weighted_choice(rng, TUBE_SAMPLING_MODE_PROBABILITIES),
        "requested_embedding_dimension": int(
            _weighted_choice(rng, REQUESTED_DIMENSION_PROBABILITIES)
        ),
    }


def _sample_num_components(rng: random.Random) -> int:
    count = 1
    p = float(DISCONNECTED_COMPONENT_CONFIG["p_add_component"])
    max_components = int(DISCONNECTED_COMPONENT_CONFIG["max_components"])
    decay = float(DISCONNECTED_COMPONENT_CONFIG["component_add_decay"])
    while count < max_components and rng.random() < p:
        count += 1
        p *= decay
    return count


def _weighted_choice(rng: random.Random, choices: tuple[tuple[Any, float], ...]) -> Any:
    total = sum(float(weight) for _, weight in choices)
    draw = rng.random() * total
    accum = 0.0
    for value, weight in choices:
        accum += float(weight)
        if draw <= accum:
            return value
    return choices[-1][0]


def _difficulty_coordinate_from_hardness(
    hardness: float,
    difficulty: str = "full",
) -> tuple[float, float, float, float]:
    preset = DIFFICULTY_PRESETS[difficulty]
    t = min(max(float(hardness), 0.0), 1.0)
    noise_ratio = _lerp(*preset["noise_ratio"], t)
    crowding_r = _lerp(*preset["crowding_r"], t)
    density = _lerp(preset["sampling_density_per_feature"][1], preset["sampling_density_per_feature"][0], t)
    thickness = _lerp(*preset["feature_thickness_ratio"], t)
    return float(noise_ratio), float(crowding_r), float(density), float(thickness)


def _lerp(low: float, high: float, t: float) -> float:
    return (1.0 - t) * float(low) + t * float(high)


def _strip_large_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in METADATA_EXCLUDE_KEYS
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory where the dataset will be written.")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of accepted samples to generate.")
    parser.add_argument(
        "--difficulty",
        choices=sorted(DIFFICULTY_PRESETS),
        default="full",
        help="Difficulty preset to sample from.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Master random seed.")
    parser.add_argument(
        "--epsilon-sep",
        type=float,
        default=DEFAULT_EPSILON_SEP,
        help="Structural separation floor used when estimating gamma_min(G).",
    )
    parser.add_argument(
        "--max-rejections-per-sample",
        type=int,
        default=DEFAULT_MAX_REJECTIONS_PER_SAMPLE,
        help="Maximum rejected graph attempts allowed before aborting one sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = generate_benchmark_batch(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        difficulty=args.difficulty,
        seed=args.seed,
        epsilon_sep=args.epsilon_sep,
        max_rejections_per_sample=args.max_rejections_per_sample,
    )
    print(output_path)


if __name__ == "__main__":
    main()
