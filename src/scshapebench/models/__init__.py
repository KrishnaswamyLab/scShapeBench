import torch

from scshapebench.models.mlp import MLPModel
from scshapebench.models.svm import SVMModel
from scshapebench.models.pi_svm import PISVMModel
from scshapebench.models.rf import RFModel
from scshapebench.models.gnn import GNNModel


def get_models(device: torch.device) -> list:
    return [
        GNNModel(device=device),   # connectivity readout
        MLPModel(device=device),   # topological barcode readout
        PISVMModel(),              # persistence image readout
        SVMModel(),                # summary-stats SVM readout
        RFModel(),                 # summary-stats RF readout
    ]
