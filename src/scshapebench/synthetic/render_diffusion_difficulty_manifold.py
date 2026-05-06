"""Render a diffusion-manifold view of benchmark difficulty and performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


AXES = (
    ("noise_ratio", "Noise", False),
    ("crowding_r", "Crowding", False),
    ("sampling_density_per_feature", "Density", True),
)

METRICS = (
    ("topology_score", "Topology", lambda sample: float(sample["headline_scores"]["topology_score"])),
    ("persistence_score", "Persistence", lambda sample: float(sample["headline_scores"]["persistence_score"])),
    ("cycle_recovery_score", "Cycle", lambda sample: float(sample["headline_scores"]["cycle_recovery_score"])),
    ("branch_recovery_score", "Branch", lambda sample: float(sample["headline_scores"]["branch_recovery_score"])),
)


def render_diffusion_difficulty_manifold(
    *,
    report_data: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    output_path: str | Path,
    metadata_output_path: str | Path | None = None,
    epsilon: float | None = None,
    candidate_dims: int = 12,
    axis_only: bool = False,
) -> dict[str, Any]:
    """Build diffusion coordinates from the difficulty cube and render them."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = _load_samples(report_data=report_data, report_path=report_path, manifest_path=manifest_path)
    per_sample = source["per_sample"]
    coords = np.asarray([_normalized_coord(sample["difficulty_coordinate"]) for sample in per_sample], dtype=float)

    diffusion = _compute_diffusion_geometry(
        coords=coords,
        epsilon=epsilon,
        candidate_dims=candidate_dims,
    )
    embedding = diffusion["embedding"]
    axis_fields = diffusion["axis_fields"]
    axis_directions = diffusion["axis_directions"]

    axis_plot_order = ("noise_ratio", "sampling_density_per_feature", "crowding_r")
    axis_norm = Normalize(vmin=0.0, vmax=1.0)
    axis_cmap = plt.get_cmap("plasma")
    has_metrics = bool(source["has_metrics"]) and not axis_only

    if has_metrics:
        fig = plt.figure(figsize=(18, 11))
        score_norm = Normalize(vmin=0.0, vmax=1.0)
        metric_cmap = plt.get_cmap("viridis")

        for plot_idx, (_, metric_label, metric_fn) in enumerate(METRICS, start=1):
            ax = fig.add_subplot(2, 4, plot_idx, projection="3d")
            values = np.asarray([metric_fn(sample) for sample in per_sample], dtype=float)
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                embedding[:, 2],
                c=values,
                cmap=metric_cmap,
                norm=score_norm,
                s=34,
                alpha=0.95,
                edgecolors="none",
            )
            _style_embedding_axes(ax=ax, title=f"{metric_label} on diffusion manifold")
            _draw_axis_directions(ax=ax, embedding=embedding, axis_directions=axis_directions)
            fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)

        for plot_idx, axis_key in enumerate(axis_plot_order, start=5):
            ax = fig.add_subplot(2, 4, plot_idx, projection="3d")
            values = axis_fields[axis_key]
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                embedding[:, 2],
                c=values,
                cmap=axis_cmap,
                norm=axis_norm,
                s=34,
                alpha=0.95,
                edgecolors="none",
            )
            _style_embedding_axes(ax=ax, title=f"{_axis_label(axis_key)} scalar field")
            _draw_axis_directions(ax=ax, embedding=embedding, axis_directions=axis_directions)
            fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)

        info_ax = fig.add_subplot(2, 4, 8)
        info_ax.axis("off")
        info_ax.text(
            0.0,
            1.0,
            _format_metadata_summary(diffusion, source),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )

        fig.suptitle(
            "Diffusion Manifold of the Difficulty Cube\n"
            "Top: performance on the matched diffusion embedding. Bottom: difficulty-axis scalar fields and matched directions.",
            fontsize=16,
            y=0.98,
        )
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.92, wspace=0.12, hspace=0.16)
    else:
        fig = plt.figure(figsize=(16, 7.4))
        for plot_idx, axis_key in enumerate(axis_plot_order, start=1):
            ax = fig.add_subplot(1, 3, plot_idx, projection="3d")
            values = axis_fields[axis_key]
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                embedding[:, 2],
                c=values,
                cmap=axis_cmap,
                norm=axis_norm,
                s=34,
                alpha=0.95,
                edgecolors="none",
            )
            _style_embedding_axes(ax=ax, title=f"{_axis_label(axis_key)} colored manifold")
            fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.11)

        fig.suptitle(
            "Difficulty-Coordinate Diffusion Manifold\n"
            f"{source['sample_count']} samples colored by noise, density hardness, and crowding.",
            fontsize=15,
            y=0.955,
        )
        fig.subplots_adjust(left=0.03, right=0.96, bottom=0.06, top=0.82, wspace=0.24)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    metadata = {
        "report_path": str(Path(report_path).resolve()) if report_path is not None else None,
        "manifest_path": str(Path(manifest_path).resolve()) if manifest_path is not None else None,
        "output_path": str(output_path.resolve()),
        "sample_count": int(source["sample_count"]),
        "has_metrics": bool(source["has_metrics"]),
        "axis_only": bool(not has_metrics),
        "epsilon": float(diffusion["epsilon"]),
        "selected_eigenvectors": diffusion["selected_eigenvectors"],
        "candidate_eigenvalues": diffusion["candidate_eigenvalues"],
        "axis_correlations": diffusion["axis_correlations"],
        "axis_regression_r2": diffusion["axis_regression_r2"],
    }
    if metadata_output_path is not None:
        metadata_output_path = Path(metadata_output_path)
        metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_output_path.write_text(json.dumps(_to_jsonable(metadata), indent=2) + "\n", encoding="utf-8")
    return metadata


def _load_samples(
    *,
    report_data: dict[str, Any] | None,
    report_path: str | Path | None,
    manifest_path: str | Path | None,
) -> dict[str, Any]:
    provided = int(report_data is not None) + int(report_path is not None) + int(manifest_path is not None)
    if provided != 1:
        raise ValueError("Provide exactly one of report_data, report_path, or manifest_path.")

    if report_data is not None:
        per_sample = list(report_data["per_sample"])
        return {
            "per_sample": per_sample,
            "has_metrics": all("headline_scores" in sample for sample in per_sample),
            "sample_count": len(per_sample),
            "source_kind": "report_data",
        }

    if report_path is not None:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        per_sample = list(report["per_sample"])
        return {
            "per_sample": per_sample,
            "has_metrics": all("headline_scores" in sample for sample in per_sample),
            "sample_count": len(per_sample),
            "source_kind": "report",
        }

    per_sample = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            per_sample.append(json.loads(stripped))
    per_sample.sort(key=lambda sample: int(sample["sample_id"]))
    return {
        "per_sample": per_sample,
        "has_metrics": False,
        "sample_count": len(per_sample),
        "source_kind": "manifest",
    }


def _compute_diffusion_geometry(
    *,
    coords: np.ndarray,
    epsilon: float | None,
    candidate_dims: int,
) -> dict[str, Any]:
    pairwise_sq = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    nonzero_sq = pairwise_sq[pairwise_sq > 0.0]
    resolved_epsilon = float(epsilon) if epsilon is not None else float(np.median(nonzero_sq))
    if resolved_epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    affinity = np.exp(-pairwise_sq / resolved_epsilon)
    degree = np.sum(affinity, axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    symmetric = inv_sqrt_degree[:, None] * affinity * inv_sqrt_degree[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)

    max_candidates = min(candidate_dims + 1, eigenvectors.shape[1])
    candidate_indices = list(range(1, max_candidates))
    candidate_vectors = eigenvectors[:, candidate_indices] * eigenvalues[candidate_indices]

    axis_fields: dict[str, np.ndarray] = {}
    selected_eigenvectors: dict[str, int] = {}
    axis_correlations: dict[str, dict[str, float]] = {}
    used_cols: set[int] = set()

    for axis_key, _, invert in AXES:
        values = coords[:, [axis_idx for axis_idx, (key, _, _) in enumerate(AXES) if key == axis_key][0]].copy()
        axis_fields[axis_key] = values

        best = None
        per_axis_corrs: dict[str, float] = {}
        for col_idx, candidate_index in enumerate(candidate_indices):
            corr = _safe_corr(candidate_vectors[:, col_idx], values)
            per_axis_corrs[str(candidate_index)] = float(corr)
            if col_idx in used_cols:
                continue
            score = abs(corr)
            if best is None or score > best[0]:
                best = (score, col_idx, candidate_index, corr)
        if best is None:
            raise RuntimeError(f"Unable to match a diffusion eigenvector for axis {axis_key}.")
        _, chosen_col, chosen_index, chosen_corr = best
        used_cols.add(chosen_col)
        selected = candidate_vectors[:, chosen_col].copy()
        if chosen_corr < 0:
            selected *= -1.0
        candidate_vectors[:, chosen_col] = selected
        selected_eigenvectors[axis_key] = int(chosen_index)
        axis_correlations[axis_key] = per_axis_corrs

    embedding = np.column_stack(
        [
            candidate_vectors[:, candidate_indices.index(selected_eigenvectors["noise_ratio"])],
            candidate_vectors[:, candidate_indices.index(selected_eigenvectors["crowding_r"])],
            candidate_vectors[:, candidate_indices.index(selected_eigenvectors["sampling_density_per_feature"])],
        ]
    )
    embedding = _normalize_embedding(embedding)

    axis_directions: dict[str, np.ndarray] = {}
    axis_regression_r2: dict[str, float] = {}
    ones = np.ones((embedding.shape[0], 1), dtype=float)
    design = np.concatenate([ones, embedding], axis=1)
    for axis_key, _, _ in AXES:
        target = axis_fields[axis_key]
        coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        direction = np.asarray(coef[1:], dtype=float)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        axis_directions[axis_key] = direction
        predicted = design @ coef
        ss_res = float(np.sum((target - predicted) ** 2))
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        axis_regression_r2[axis_key] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "epsilon": resolved_epsilon,
        "embedding": embedding,
        "axis_fields": axis_fields,
        "selected_eigenvectors": selected_eigenvectors,
        "candidate_eigenvalues": {
            str(candidate_index): float(eigenvalues[candidate_index])
            for candidate_index in candidate_indices
        },
        "axis_correlations": axis_correlations,
        "axis_directions": axis_directions,
        "axis_regression_r2": axis_regression_r2,
    }


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    centered = embedding - embedding.mean(axis=0, keepdims=True)
    scale = np.max(np.abs(centered), axis=0, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return centered / scale


def _style_embedding_axes(*, ax, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Noise Axis", fontsize=9, labelpad=8)
    ax.set_ylabel("Crowding Axis", fontsize=9, labelpad=8)
    ax.set_zlabel("Density Axis", fontsize=9, labelpad=8)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=21, azim=-56)
    ax.grid(False)


def _draw_axis_directions(*, ax, embedding: np.ndarray, axis_directions: dict[str, np.ndarray]) -> None:
    center = embedding.mean(axis=0)
    lengths = np.max(np.abs(embedding - center), axis=0)
    scale = 0.55 * float(np.mean(lengths))
    colors = {
        "noise_ratio": "#ef4444",
        "crowding_r": "#2563eb",
        "sampling_density_per_feature": "#059669",
    }
    labels = {
        "noise_ratio": "noise+",
        "crowding_r": "crowd+",
        "sampling_density_per_feature": "dense-hard+",
    }
    for axis_key, direction in axis_directions.items():
        delta = scale * direction
        ax.quiver(
            center[0],
            center[1],
            center[2],
            delta[0],
            delta[1],
            delta[2],
            color=colors[axis_key],
            linewidth=2.2,
            arrow_length_ratio=0.12,
        )
        tip = center + delta
        ax.text(tip[0], tip[1], tip[2], labels[axis_key], color=colors[axis_key], fontsize=8)


def _format_metadata_summary(diffusion: dict[str, Any], source: dict[str, Any]) -> str:
    lines = [
        "Matched Diffusion Geometry",
        "",
        f"samples = {source['sample_count']}",
        f"source = {source['source_kind']}",
        "",
        f"epsilon = {diffusion['epsilon']:.4f}",
        "",
        "selected eigenvectors:",
    ]
    for axis_key, axis_label, _ in AXES:
        lines.append(f"  {axis_label:<9} -> phi_{diffusion['selected_eigenvectors'][axis_key]}")
    lines.append("")
    lines.append("axis regression R^2:")
    for axis_key, axis_label, _ in AXES:
        lines.append(f"  {axis_label:<9} {diffusion['axis_regression_r2'][axis_key]:.3f}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  arrows show the manifold directions")
    lines.append("  along which each original difficulty")
    lines.append("  axis increases most strongly.")
    return "\n".join(lines)


def _axis_label(axis_key: str) -> str:
    labels = {
        "noise_ratio": "Noise",
        "crowding_r": "Crowding",
        "sampling_density_per_feature": "Density Hardness",
    }
    return labels[axis_key]


def _normalized_coord(difficulty_coordinate: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            _normalize_interval(float(difficulty_coordinate["noise_ratio"]), low=0.02, high=0.20, invert=False),
            _normalize_interval(float(difficulty_coordinate["crowding_r"]), low=0.00, high=0.60, invert=False),
            _normalize_interval(
                float(difficulty_coordinate["sampling_density_per_feature"]),
                low=3.5,
                high=25.0,
                invert=True,
            ),
        ],
        dtype=float,
    )


def _normalize_interval(value: float, *, low: float, high: float, invert: bool) -> float:
    clipped = min(max(float(value), low), high)
    fraction = (clipped - low) / (high - low)
    if invert:
        fraction = 1.0 - fraction
    return float(min(max(fraction, 0.0), 1.0))


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std < 1e-12 or right_std < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


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
    parser.add_argument(
        "--report-path",
        default=None,
        help="Evaluator report JSON to visualize.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Manifest JSONL to visualize directly when no evaluation report is available.",
    )
    parser.add_argument("--output-path", required=True, help="Image path to write.")
    parser.add_argument(
        "--metadata-output-path",
        default=None,
        help="Optional JSON path to write embedding metadata.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Optional Gaussian affinity scale. Defaults to median pairwise squared distance.",
    )
    parser.add_argument(
        "--candidate-dims",
        type=int,
        default=12,
        help="How many nontrivial diffusion eigenvectors to consider for axis matching.",
    )
    parser.add_argument(
        "--axis-only",
        action="store_true",
        help="Render only the three axis-colored manifold views, even when report metrics are present.",
    )
    args = parser.parse_args()
    if (args.report_path is None) == (args.manifest_path is None):
        parser.error("Provide exactly one of --report-path or --manifest-path.")
    return args


def main() -> None:
    args = _parse_args()
    summary = render_diffusion_difficulty_manifold(
        report_path=args.report_path,
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        metadata_output_path=args.metadata_output_path,
        epsilon=args.epsilon,
        candidate_dims=args.candidate_dims,
        axis_only=args.axis_only,
    )
    print(json.dumps(_to_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
