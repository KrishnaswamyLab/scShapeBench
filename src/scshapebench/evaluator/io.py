import json
from datetime import datetime
from pathlib import Path


def store_result(
    all_results: dict,
    model_name: str,
    fold_id: str,
    metrics: dict,
) -> None:
    if model_name not in all_results:
        all_results[model_name] = {}
    all_results[model_name][fold_id] = metrics


def save_results(final_results: dict, output_dir: Path, metadata: dict) -> Path:
    output = {"metadata": metadata, "results": final_results}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"results_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    return path
