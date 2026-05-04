import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from scshapebench.datasets.labels import CLASS_ORDER

try:
    from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence
except ImportError:
    ArpackNoConvergence = Exception

LPE_DIM = 8
NODE_FEATURES = 2 + LPE_DIM  # (x, y) + LPE eigenvectors
NUM_CLASSES = len(CLASS_ORDER)  # Strictly locked to 


class _GraphData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        return 1 if key == "edge_index" else 0

    def __inc__(self, key, value, *args, **kwargs):
        return self.x.size(0) if key == "edge_index" else super().__inc__(key, value, *args, **kwargs)


def _dense_lpe(n_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor, k: int) -> torch.Tensor:
    L = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    w   = edge_weight.numpy() if edge_weight.numel() > 0 else np.ones(len(src))
    for i, j, wij in zip(src, dst, w):
        L[i, j] -= wij
        L[i, i] += wij
    L = (L + L.T) / 2
    _, eig_vecs = np.linalg.eigh(L)
    pe = eig_vecs[:, 1:k + 1].astype(np.float32)
    if pe.shape[1] < k:
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    return torch.from_numpy(pe)


def _prepare_graph(g: Data, y: np.ndarray) -> _GraphData:
    n_nodes = g.x.size(0)
    k_eff   = min(LPE_DIM, n_nodes - 1)

    if k_eff > 0:
        try:
            lpe = AddLaplacianEigenvectorPE(k=k_eff, attr_name="pe", is_undirected=True)
            tmp = lpe(Data(x=g.x, edge_index=g.edge_index, edge_weight=g.edge_weight))
            pe  = tmp.pe
        except ArpackNoConvergence:
            pe = _dense_lpe(n_nodes, g.edge_index, g.edge_weight, k_eff)
        if k_eff < LPE_DIM:
            pe = F.pad(pe, (0, LPE_DIM - k_eff))
    else:
        pe = torch.zeros(n_nodes, LPE_DIM)

    x = torch.cat([g.x, pe], dim=1)
    out = _GraphData(x=x, edge_index=g.edge_index.clone(), edge_weight=g.edge_weight.clone())
    
    # Enforce strict (1, 6) shape so PyG batches it to (batch_size, 6)
    out.y = torch.tensor(y, dtype=torch.float).view(1, NUM_CLASSES)
    return out


class _GraphBranch(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden)
        self.bn   = BatchNorm1d(hidden)

    def forward(self, x, edge_index, edge_weight, batch):
        return global_mean_pool(self.bn(self.conv(x, edge_index, edge_weight).relu()), batch)


class _MultiLabelGraphGNN(torch.nn.Module):
    def __init__(self, in_channels: int = NODE_FEATURES, hidden: int = 64):
        super().__init__()
        self.branch = _GraphBranch(in_channels, hidden)
        self.lin1   = Linear(hidden, hidden)
        self.lin2   = Linear(hidden, NUM_CLASSES)  # Strictly outputs 6

    def forward(self, data) -> torch.Tensor:
        h = self.branch(data.x, data.edge_index, data.edge_weight, data.batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        return self.lin2(h)


class GNNModel:
    name = "gnn"

    def __init__(self, device=None, hidden: int = 64, batch_size: int = 16,
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

        in_channels = data_list[0].x.shape[1]
        
        self.model = _MultiLabelGraphGNN(in_channels, self.hidden).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            self.model.train()
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch), batch.y)
                loss.backward()
                optimizer.step()

    def predict(self, graphs, features=None):
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
            
        # Dummy label array of shape (6,) to satisfy _prepare_graph during inference
        dummy_label = np.zeros(NUM_CLASSES, dtype=np.float32)
        data_list = [_prepare_graph(g, dummy_label) for g in graphs]
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for batch in loader:
                all_logits.append(self.model(batch.to(self.device)).cpu())

        logits = torch.cat(all_logits).numpy()  # Will be shape (N, 6)
        probs  = 1 / (1 + np.exp(-logits))
        preds  = (probs > 0.5).astype(int)

        return preds, probs