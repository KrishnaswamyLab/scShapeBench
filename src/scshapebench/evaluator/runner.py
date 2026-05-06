"""Shared real-data benchmark runner.

This module centralizes the real-track evaluation flow so both existing
entrypoints can share the same implementation without duplicating split,
feature, baseline, and model-evaluation logic.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import baseline_predictions
from .io import save_results, store_result
from .loader import load_graphs
from .metrics import aggregate, evaluate
from .validator import validate_submission
from ..datasets.labels import load_labels
from ..features.graph_stats import extract_features as extract_graph_features
from ..features.persistence import extract_features as extract_persistence_features
from ..models import get_models

LABEL_PATHS = {
    "majority": "benchmarks/scrnaseq/labels/majority_labels.json",
    "union": "benchmarks/scrnaseq/labels/union_labels.json",
    "confidence_score": "benchmarks/scrnaseq/labels/confidence_labels.json",
    "high_support": "benchmarks/scrnaseq/labels/high_support_labels.json",
}


def make_folds(graphs: dict[str, Any], k: int = 5, seed: int = 42) -> list[dict[str, list[str]]]:
    ids = sorted(graphs.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    folds = []
    for i in range(k):
        test_ids = set(ids[i::k])
        train_ids = [x for x in ids if x not in test_ids]
        folds.append({"train": train_ids, "test": list(test_ids)})
    return folds


def load_folds(splits_path: Path) -> list[dict[str, list[str]]]:
    data = json.loads(splits_path.read_text())
    return [{"train": data[key]["train"], "test": data[key]["test"]} for key in sorted(data.keys())]


def filter_split_ids(ids: list[str], available_ids: set[str], split_name: str) -> list[str]:
    kept = [sid for sid in ids if sid in available_ids]
    dropped = sorted(set(ids) - available_ids)
    if dropped:
        preview = ", ".join(dropped[:5])
        suffix = "..." if len(dropped) > 5 else ""
        print(f"  Dropped {len(dropped)} {split_name} ID(s) without both graph and label: {preview}{suffix}")
    return kept


def set_model_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_real_submission(
    submission_dir: Path | str,
    *,
    label_type: str,
    output_dir: Path | str,
    device: torch.device | str,
    splits_path: Path | None = None,
    fold_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    submission_dir = Path(submission_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    graphs, metadata = load_graphs(submission_dir)
    validate_submission(graphs)

    labels, _, dataset_order = load_labels(path=LABEL_PATHS[label_type])
    id_to_idx = {sid: i for i, sid in enumerate(dataset_order)}
    evaluable_ids = set(graphs) & set(dataset_order)
    graph_only = sorted(set(graphs) - set(dataset_order))
    label_only = sorted(set(dataset_order) - set(graphs))
    if graph_only:
        print(f"Warning: excluding {len(graph_only)} graph(s) with no label: {', '.join(graph_only[:5])}")
    if label_only:
        print(f"Warning: excluding {len(label_only)} label(s) with no graph: {', '.join(label_only[:5])}")
    if not evaluable_ids:
        raise ValueError("No dataset IDs have both graphs and labels.")

    if splits_path is not None:
        print(f"Using predefined splits from {splits_path}")
        splits = load_folds(splits_path)
    else:
        print("Using generated folds over evaluable graph/label IDs")
        splits = make_folds({sid: graphs[sid] for sid in evaluable_ids}, k=5, seed=fold_seed)

    all_results: dict[str, Any] = {}
    for fold_id, split in enumerate(splits):
        train_ids = filter_split_ids(split["train"], evaluable_ids, "train")
        test_ids = filter_split_ids(split["test"], evaluable_ids, "test")

        train_graphs = [graphs[i] for i in train_ids]
        test_graphs = [graphs[i] for i in test_ids]
        train_labels = labels[[id_to_idx[i] for i in train_ids]]
        test_labels = labels[[id_to_idx[i] for i in test_ids]]

        if len(train_graphs) != len(train_labels) or len(test_graphs) != len(test_labels):
            raise RuntimeError("Internal ID alignment error between graphs and labels.")

        features_cache = {
            "persistence": extract_persistence_features(train_graphs, test_graphs),
            "graph_stats": extract_graph_features(train_graphs, test_graphs),
        }

        for baseline_name, (preds, probs) in baseline_predictions(
            train_labels=train_labels,
            test_labels=test_labels,
            seed=20_000 + fold_id,
        ).items():
            store_result(
                all_results,
                model_name=baseline_name,
                fold_id=fold_id,
                metrics=evaluate(test_labels, preds, probs),
            )

        for model_idx, model in enumerate(get_models(device)):
            set_model_seed(10_000 + fold_id * 100 + model_idx)
            src = model.features_source
            X_train_m, X_test_m = features_cache[src] if src else (None, None)
            model.fit(graphs=train_graphs, labels=train_labels, features=X_train_m)
            preds, probs = model.predict(graphs=test_graphs, features=X_test_m)
            store_result(
                all_results,
                model_name=model.name,
                fold_id=fold_id,
                metrics=evaluate(test_labels, preds, probs),
            )

    final_results = aggregate(all_results)
    save_results(final_results, output_dir, metadata)
    return final_results, metadata


def run_real_collection(
    submissions_root: Path | str,
    *,
    label_type: str,
    output_dir: Path | str,
    device: torch.device | str,
    methods: list[str] | None = None,
    splits_path: Path | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    submissions_root = Path(submissions_root)
    if not submissions_root.exists():
        raise FileNotFoundError(f"Submissions directory not found: {submissions_root}")

    method_dirs = sorted(
        d for d in submissions_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if methods:
        method_dirs = [d for d in method_dirs if d.name in methods]

    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for method_dir in method_dirs:
        results.append(
            run_real_submission(
                method_dir,
                label_type=label_type,
                output_dir=output_dir,
                device=device,
                splits_path=splits_path,
            )
        )
    return results
