import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from scshapebench.datasets.labels import CLASS_ORDER


class RFModel:
    """Multi-label Random Forest on hand-crafted graph-stat features (RF readout)."""

    name            = "rf"
    features_source = "graph_stats"

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._classifiers: dict = {}

    def fit(self, graphs, labels: np.ndarray, features: np.ndarray) -> None:
        self._classifiers = {}
        for cls_idx, cls_name in enumerate(CLASS_ORDER):
            y = labels[:, cls_idx].astype(int)

            if len(np.unique(y)) < 2:
                self._classifiers[cls_name] = None
                continue

            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            clf.fit(features, y)
            self._classifiers[cls_name] = clf

    def predict(self, graphs, features: np.ndarray):
        n = len(features)
        preds = np.zeros((n, len(CLASS_ORDER)), dtype=int)
        probs = np.zeros((n, len(CLASS_ORDER)), dtype=np.float32)

        for cls_idx, cls_name in enumerate(CLASS_ORDER):
            clf = self._classifiers.get(cls_name)
            if clf is None:
                continue
            preds[:, cls_idx] = clf.predict(features)
            probs[:, cls_idx] = clf.predict_proba(features)[:, 1]

        return preds, probs
