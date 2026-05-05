import numpy as np

from scshapebench.datasets.labels import CLASS_ORDER


def baseline_predictions(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return deterministic multi-label baseline predictions for one fold."""
    n_test = test_labels.shape[0]
    n_classes = len(CLASS_ORDER)
    prevalence = train_labels.mean(axis=0).astype(np.float32)

    zeros = np.zeros((n_test, n_classes), dtype=np.float32)
    ones = np.ones((n_test, n_classes), dtype=np.float32)
    prevalence_probs = np.tile(prevalence, (n_test, 1))

    rng = np.random.default_rng(seed)
    random_probs = rng.random((n_test, n_classes), dtype=np.float32)
    random_preds = (random_probs < prevalence).astype(int)

    return {
        "baseline_all_zero": (zeros.astype(int), zeros),
        "baseline_all_one": (ones.astype(int), ones),
        "baseline_train_prevalence": ((prevalence_probs >= 0.5).astype(int), prevalence_probs),
        "baseline_random_prevalence": (random_preds, random_probs),
    }
