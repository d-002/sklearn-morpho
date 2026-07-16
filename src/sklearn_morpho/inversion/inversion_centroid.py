import numpy as np

from .inversion_base import InversionHeuristic


class CentroidInversion(InversionHeuristic):
    """
Let $c0$, $c1$ be the centroids of the negative and positive classes
respectively, and $vector$ the given initialization parameter.
This heuristic performs an inversion when $c1-c0 \\cdot vector < 0$.

Users may then provide an optimal placement of classes centroids, and if the
dataset is organized in the opposite manner then an inversion will take place.
    """

    def __init__(self, perfect_vector: np.ndarray) -> None:
        self.perfect_vector = perfect_vector

    def should_invert(self, X: np.ndarray, y: np.ndarray) -> bool:
        s0, s1 = X.shape[1:], self.perfect_vector.shape
        if s0 != s1:
            raise ValueError(
                'Incompatible data passed to should_invert: '
                f'expected size {s1} but got {s0}'
            )

        labels, inv, counts = np.unique(
            y, return_inverse=True, return_counts=True
        )
        sums = np.zeros((len(labels), X.shape[1]))
        np.add.at(sums, inv, X)
        centroids = sums / counts[:, np.newaxis]

        dataset_vector = centroids[1] - centroids[0]

        res: bool = dataset_vector @ self.perfect_vector < 0
        return res
