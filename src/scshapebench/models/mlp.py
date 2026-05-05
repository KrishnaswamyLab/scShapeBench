import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scshapebench.datasets.labels import CLASS_ORDER
from scshapebench.features.persistence import FEAT_DIM


_NUM_CLASSES = 4


class _MultiMLP(torch.nn.Module):
    def __init__(self, in_features: int = FEAT_DIM, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, _NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        criterion(model(X_batch.to(device)), y_batch.to(device)).backward()
        optimizer.step()


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    all_logits = []
    for X_batch, _ in loader:
        all_logits.append(model(X_batch.to(device)).cpu())
    logits = torch.cat(all_logits).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    return preds, probs


class MLPModel:
    name            = "mlp"
    features_source = "persistence"

    def __init__(self, device=None, hidden: int = 64, batch_size: int = 16,
                 lr: float = 1e-3, epochs: int = 1):
        self.device = device or torch.device("cpu")
        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self._state = None  # (mean, std, model)

    def fit(self, graphs, labels: np.ndarray, features: np.ndarray) -> None:
        mean = features.mean(axis=0)
        std  = features.std(axis=0) + 1e-8
        X    = ((features - mean) / std).astype(np.float32)
        y    = labels.astype(np.float32)

        loader = _make_loader(X, y, self.batch_size, shuffle=True)
        model = _MultiMLP(in_features=X.shape[1], hidden=self.hidden).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            _train_epoch(model, loader, optimizer, criterion, self.device)

        self._state = (mean, std, model)

    def predict(self, graphs, features: np.ndarray):
        mean, std, model = self._state
        X = ((features - mean) / std).astype(np.float32)
        dummy_y = np.zeros((len(X), _NUM_CLASSES), dtype=np.float32)
        loader = _make_loader(X, dummy_y, self.batch_size, shuffle=False)
        return _predict(model, loader, self.device)
