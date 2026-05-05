import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scshapebench.datasets.labels import CLASS_ORDER


class PISVMModel:
    """Linear-kernel SVM on H0+H1 persistence image features (PI-SVM readout)."""

    name            = "pi_svm"
    features_source = "persistence"

    def __init__(self, C: float = 1.0):
        self.C = C
        self._classifiers: dict = {}

    def fit(self, graphs, labels: np.ndarray, features: np.ndarray) -> None:
        self._classifiers = {}
        for cls_idx, cls_name in enumerate(CLASS_ORDER):
            y = labels[:, cls_idx].astype(int)

            if len(np.unique(y)) < 2:
                self._classifiers[cls_name] = None
                continue

            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm",    SVC(C=self.C, kernel="linear", probability=True)),
            ])
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
