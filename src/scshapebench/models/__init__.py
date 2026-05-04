import torch

from scshapebench.models.mlp import MLPModel
from scshapebench.models.svm import SVMModel
from scshapebench.models.gnn import GNNModel


def get_models(device: torch.device) -> list:
    return [
        MLPModel(device=device),
        SVMModel(),
        GNNModel(device=device),
    ]
