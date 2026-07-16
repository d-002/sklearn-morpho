import numpy as np


def friendly_dataset(
    n_samples: int = 100, random_state: np.random.RandomState | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a toy 2D dataset that is suitable for all l-DEP, DEP and simple
    morphological perceptrons.
    """

    centroids = np.array([[-5, -3], [3, 2]])
    std = np.array([1.5, 1])

    if random_state is None:
        random_state = np.random.RandomState()

    n0 = n_samples // 2
    n1 = n_samples - n0

    X0 = random_state.randn(n0, 2) * std[0] + centroids[0]
    X1 = random_state.randn(n1, 2) * std[1] + centroids[1]
    y0, y1 = np.zeros(n0, dtype=int), np.ones(n1, dtype=int)

    X = np.concatenate([X0, X1])
    y = np.concatenate([y0, y1])

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    return X[indices], y[indices]
