from __future__ import annotations

import numpy as np

from . import SampleWeighting

class DistSampleWeighting(SampleWeighting):
    """
    Weighting method that weights its inputs inversely proportionally to the
    distance to their class' centroid.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> DistSampleWeighting:
        # compute centroids, there should be no classes with no elements thanks
        # to sklearn checks if used in the intended way with a compatible
        # estimator
        labels, inv, counts = np.unique(y, return_inverse=True,
                                        return_counts=True)
        sums = np.zeros((len(labels), X.shape[1]))
        np.add.at(sums, inv, X)
        centroids = sums / counts[:, np.newaxis]

        # inverse distance from each data point to its respective class centroid
        self.weights_ = 1 / (1e-9 + np.linalg.norm(X - centroids[inv], axis=1))
        max_centroid_w = np.zeros(len(labels))
        np.maximum.at(max_centroid_w, inv, self.weights_)
        self.weights_ /= max_centroid_w[y]

        cost_normalizer = self.weights_.sum() / len(labels)
        self.cost_normalizer_ = cost_normalizer

        return self
