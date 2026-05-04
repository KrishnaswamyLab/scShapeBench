import json


def load_splits(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
