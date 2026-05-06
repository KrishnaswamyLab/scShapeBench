"""Compatibility facade for the modular cycle-backbone generation pipeline."""

from __future__ import annotations

from .backbone import *  # noqa: F401,F403
from .backbone import __all__ as _backbone_all
from .branching import *  # noqa: F401,F403
from .branching import __all__ as _branching_all
from .embedding import *  # noqa: F401,F403
from .embedding import __all__ as _embedding_all
from .geometry2d import *  # noqa: F401,F403
from .geometry2d import __all__ as _geometry2d_all
from .geometry3d import *  # noqa: F401,F403
from .geometry3d import __all__ as _geometry3d_all
from .graph_primitives import *  # noqa: F401,F403
from .graph_primitives import __all__ as _graph_primitives_all
from .metadata_utils import *  # noqa: F401,F403
from .metadata_utils import __all__ as _metadata_utils_all
from .point_cloud import *  # noqa: F401,F403
from .point_cloud import __all__ as _point_cloud_all

__all__ = list(
    dict.fromkeys(
        [
            *_graph_primitives_all,
            *_metadata_utils_all,
            *_backbone_all,
            *_branching_all,
            *_geometry2d_all,
            *_geometry3d_all,
            *_embedding_all,
            *_point_cloud_all,
        ]
    )
)


if __name__ == "__main__":
    import networkx as nx

    accepted = False
    demo_rejections = 0
    for offset in range(20):
        backbone_seed = 29 + offset
        branch_seed = 314 + offset
        embedding_seed = 2718 + offset
        point_cloud_seed = 1618 + offset

        G, meta = generate_cycle_backbone(
            num_cycles=5,
            cycle_len_range=(4, 8),
            sep_range=(-1, 3),
            seed=backbone_seed,
        )
        betti_before = betti_1(G)

        G, meta = add_random_branches(
            G=G,
            branch_attach_count_range=(1, 4),
            children_per_branch_point_range=(1, 3),
            branch_length_range=(1, 4),
            max_branch_depth_range=(1, 3),
            promotion_prob_range=(0.2, 0.6),
            max_degree=5,
            seed=branch_seed,
            metadata=meta,
        )

        betti_after = betti_1(G)
        pos, meta = embed_graph_constructively(
            G=G,
            metadata=meta,
            feature_scale_range=(1.0, 1.0),
            separation_to_feature_ratio_range=(0.8, 2.0),
            noise_to_feature_ratio_range=(0.02, 0.25),
            sampling_density_per_feature_range=(10.0, 40.0),
            edge_length_jitter_ratio_range=(0.0, 0.12),
            branch_angle_spread_range=(0.5, 1.2),
            max_local_placement_attempts=20,
            max_global_resampling_attempts=5,
            max_embedding_dimension=5,
            seed=embedding_seed,
        )
        if meta["embedding_rejected"]:
            demo_rejections += 1
            continue
        accepted = True
        break

    if not accepted:
        raise RuntimeError("Constructive embedding failed in both 2D and 3D for all demo retries.")
    clean_points, noisy_points, meta = sample_noisy_point_cloud_from_embedding(
        G=G,
        pos=pos,
        metadata=meta,
        include_node_points=True,
        min_points_per_edge=2,
        density_jitter_ratio_range=(0.0, 0.15),
        endpoint_sampling_mode="half-open",
        seed=point_cloud_seed,
    )

    print(
        f"backbone_seed={backbone_seed} branch_seed={branch_seed} "
        f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
        f"betti_before={betti_before} betti_after={betti_after} "
        f"demo_rejections_before_accept={demo_rejections}"
    )
    print(
        f"leaf_count={leaf_count(G)} branch_node_count={len(branch_nodes(G))} "
        f"planar={meta['is_planar']}"
    )
    print(f"branch_params={meta['branch_params']}")
    print(f"branch_attachment_nodes_initial={meta['branch_attachment_nodes_initial']}")
    print(f"branch_generation_nodes={meta['branch_generation_nodes']}")
    print(f"embedding_params={meta['embedding_params']}")
    print(
        f"embedding_valid={meta['embedding_valid']} "
        f"embedding_dimension={meta['embedding_dimension']} "
        f"min_node_sep={meta['min_nonadjacent_node_distance']:.3f} "
        f"min_edge_sep={meta['min_nonincident_edge_distance']:.3f}"
    )
    print(
        f"num_clean_points={meta['num_clean_points']} "
        f"sigma={meta['point_cloud_params']['sigma']:.4f} "
        f"density={meta['point_cloud_params']['sampling_density_per_feature']:.3f} "
        f"feature_scale={meta['point_cloud_params']['feature_scale']:.3f}"
    )

    sample_parent_items = list(meta["branch_parent_map"].items())[:10]
    sample_root_items = list(meta["branch_root_map"].items())[:10]
    print(f"branch_parent_map_sample={sample_parent_items}")
    print(f"branch_root_map_sample={sample_root_items}")

    assert nx.is_connected(G)
    assert betti_before == betti_after
    assert len(pos) == G.number_of_nodes()
    del clean_points, noisy_points
