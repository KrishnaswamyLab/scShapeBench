import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool
from torch_geometric.utils import degree

from scshapebench.datasets.labels import CLASS_ORDER

NUM_CLASSES = len(CLASS_ORDER)
NODE_FEAT_DIM = 7
EDGE_FEAT_DIM = 2


def _prepare_graph(g: Data, y: np.ndarray) -> Data:
    n_nodes = g.x.size(0)
    deg = degree(g.edge_index[0], num_nodes=n_nodes).float()
    deg_norm = deg / deg.max().clamp(min=1.0)

    coords = g.x.float()
    if coords.size(1) >= 2:
        xy = coords[:, :2]
    elif coords.size(1) == 1:
        xy = torch.cat([coords, torch.zeros_like(coords)], dim=1)
    else:
        xy = torch.zeros((n_nodes, 2), dtype=torch.float)

    xy_mean = xy.mean(dim=0, keepdim=True)
    xy_std = xy.std(dim=0, keepdim=True).clamp(min=1e-6)
    xy_norm = (xy - xy_mean) / xy_std

    edge_index = g.edge_index.clone()
    src = edge_index[0]
    dst = edge_index[1]
    pair_counts: dict[tuple[int, int], int] = {}
    adjacency = [set() for _ in range(n_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            adjacency[u].add(v)
            key = (min(u, v), max(u, v))
            pair_counts[key] = pair_counts.get(key, 0) + 1

    pair_multiplicity = {
        key: max(count / 2.0, 1.0)
        for key, count in pair_counts.items()
    }
    max_multiplicity = max(pair_multiplicity.values(), default=1.0)

    incident_parallel_excess = torch.zeros(n_nodes, dtype=torch.float)
    for (u, v), multiplicity in pair_multiplicity.items():
        excess = max(multiplicity - 1.0, 0.0)
        incident_parallel_excess[u] += excess
        incident_parallel_excess[v] += excess
    incident_parallel_excess = incident_parallel_excess / incident_parallel_excess.max().clamp(min=1.0)

    clustering = torch.zeros(n_nodes, dtype=torch.float)
    for node, neighbors in enumerate(adjacency):
        k = len(neighbors)
        if k < 2:
            continue
        links = 0
        neighbor_list = list(neighbors)
        for i, u in enumerate(neighbor_list):
            u_neighbors = adjacency[u]
            for v in neighbor_list[i + 1:]:
                if v in u_neighbors:
                    links += 1
        clustering[node] = 2.0 * links / (k * (k - 1))

    component_size = torch.ones(n_nodes, dtype=torch.float)
    seen = [False] * n_nodes
    for start in range(n_nodes):
        if seen[start]:
            continue
        stack = [start]
        comp = []
        seen[start] = True
        while stack:
            node = stack.pop()
            comp.append(node)
            for nbr in adjacency[node]:
                if not seen[nbr]:
                    seen[nbr] = True
                    stack.append(nbr)
        frac = len(comp) / max(n_nodes, 1)
        for node in comp:
            component_size[node] = frac

    unique_degree = torch.tensor([len(neighbors) for neighbors in adjacency], dtype=torch.float)
    unique_degree = unique_degree / unique_degree.max().clamp(min=1.0)

    node_features = torch.cat([
        xy_norm,
        deg_norm.view(-1, 1),
        unique_degree.view(-1, 1),
        incident_parallel_excess.view(-1, 1),
        clustering.view(-1, 1),
        component_size.view(-1, 1),
    ], dim=1)

    edge_weight = g.edge_weight.clone() if hasattr(g, "edge_weight") else torch.zeros(edge_index.size(1))
    if edge_weight.numel() != edge_index.size(1):
        edge_weight = torch.zeros(edge_index.size(1), dtype=torch.float)

    edge_multiplicity = torch.ones(edge_index.size(1), dtype=torch.float)
    for idx, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        if u == v:
            edge_multiplicity[idx] = 1.0
            continue
        key = (min(u, v), max(u, v))
        edge_multiplicity[idx] = pair_multiplicity.get(key, 1.0) / max_multiplicity

    edge_attr = torch.stack([
        edge_weight.float(),
        edge_multiplicity.float(),
    ], dim=1)

    out = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight)
    out.y = torch.tensor(y, dtype=torch.float).view(1, NUM_CLASSES)
    return out


class _GNN(torch.nn.Module):
    def __init__(self, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GINEConv(
            Sequential(
                Linear(NODE_FEAT_DIM, hidden),
                torch.nn.ReLU(),
                Linear(hidden, hidden),
            ),
            edge_dim=EDGE_FEAT_DIM,
        )
        self.bn1 = BatchNorm1d(hidden)
        self.conv2 = GINEConv(
            Sequential(
                Linear(hidden, hidden),
                torch.nn.ReLU(),
                Linear(hidden, hidden),
            ),
            edge_dim=EDGE_FEAT_DIM,
        )
        self.bn2 = BatchNorm1d(hidden)
        self.dropout = dropout
        self.lin1 = Linear(2 * hidden, hidden)
        self.lin2 = Linear(hidden, NUM_CLASSES)

    def forward(self, data) -> torch.Tensor:
        h = self.conv1(data.x, data.edge_index, data.edge_attr)
        h = self.bn1(h).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, data.edge_index, data.edge_attr)
        h = self.bn2(h).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        pooled = torch.cat([
            global_mean_pool(h, data.batch),
            global_max_pool(h, data.batch),
        ], dim=1)
        out = self.lin1(pooled).relu()
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)


class GNNModel:
    name            = "gnn"
    features_source = None

    def __init__(self, device=None, hidden: int = 32, batch_size: int = 16,
                 lr: float = 1e-3, epochs: int = 100, dropout: float = 0.3):
        self.device     = device or torch.device("cpu")
        self.hidden     = hidden
        self.batch_size = batch_size
        self.lr         = lr
        self.epochs     = epochs
        self.dropout    = dropout
        self.model      = None

    def fit(self, graphs, labels: np.ndarray, features=None) -> None:
        data_list = [_prepare_graph(g, yi.astype(np.float32)) for g, yi in zip(graphs, labels)]
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

        self.model = _GNN(self.hidden, dropout=self.dropout).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        positives = labels.sum(axis=0).astype(np.float32)
        negatives = labels.shape[0] - positives
        pos_weight = negatives / np.clip(positives, 1.0, None)
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float, device=self.device)
        )

        for _ in range(self.epochs):
            self.model.train()
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                criterion(self.model(batch), batch.y).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
