import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import degree

from scshapebench.datasets.labels import CLASS_ORDER

NUM_CLASSES = len(CLASS_ORDER)


def _prepare_graph(g: Data, y: np.ndarray) -> Data:
    n_nodes = g.x.size(0)
    deg = degree(g.edge_index[0], num_nodes=n_nodes).unsqueeze(1)  # [N, 1]
    out = Data(x=deg, edge_index=g.edge_index.clone(), edge_weight=g.edge_weight.clone())
    out.y = torch.tensor(y, dtype=torch.float).view(1, NUM_CLASSES)
    return out


class _GNN(torch.nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.conv = GCNConv(1, hidden)
        self.lin  = Linear(hidden, NUM_CLASSES)

    def forward(self, data) -> torch.Tensor:
        h = self.conv(data.x, data.edge_index, data.edge_weight).relu()
        h = global_mean_pool(h, data.batch)
        return self.lin(h)


class GNNModel:
    name            = "gnn"
    features_source = None

    def __init__(self, device=None, hidden: int = 32, batch_size: int = 16,
                 lr: float = 1e-3, epochs: int = 50):
        self.device     = device or torch.device("cpu")
        self.hidden     = hidden
        self.batch_size = batch_size
        self.lr         = lr
        self.epochs     = epochs
        self.model      = None

    def fit(self, graphs, labels: np.ndarray, features=None) -> None:
        data_list = [_prepare_graph(g, yi.astype(np.float32)) for g, yi in zip(graphs, labels)]
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

        self.model = _GNN(self.hidden).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion  = torch.nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            self.model.train()
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                criterion(self.model(batch), batch.y).backward()
                optimizer.step()

    def predict(self, graphs, features=None):
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        dummy = np.zeros(NUM_CLASSES, dtype=np.float32)
        data_list = [_prepare_graph(g, dummy) for g in graphs]
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for batch in loader:
                all_logits.append(self.model(batch.to(self.device)).cpu())

        logits = torch.cat(all_logits).numpy()
        probs  = 1 / (1 + np.exp(-logits))
        preds  = (probs > 0.5).astype(int)
        return preds, probs
