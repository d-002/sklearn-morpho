import numpy as np

from .weighting_base import SampleWeighting

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
        self.weights_ = 1 / (1e-6 + np.linalg.norm(X - centroids[y], axis=1))
        max_centroid_w = np.array([self.weights_[y == y_].max()
                                   for y_ in range(2)])
        self.weights_ /= max_centroid_w[y]

        cost_normalizer = self.weights_.sum()
        if np.isclose(cost_normalizer, 0):
            self.cost_normalizer_ = 1
        else:
            self.cost_normalizer_ = cost_normalizer

        return self
