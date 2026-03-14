import numpy as np
from random import random

class SampleData:
    """Sample data creator, for data classification using a single perceptron.
    Creates randomly generated data inside a box bounded by min and max.
    Every element in the data is part of either class 0, or class 1, and class 0
    is made of the quadrant lower than quadrant_bound, with additional noise.

    The data is stored in X and Y, X being the points and Y their respective
    classes, for easy integration with scikit-learn.
    """

    def __init__(self, mean: np.ndarray, deviation: np.ndarray,
                 quadrant_bound: np.ndarray, n: int, noise: float) -> None:
        assert mean.shape[0] == deviation.shape[0]

        normal = lambda *args: np.multiply(np.random.randn(*args), deviation) \
                    + mean

        self.X = normal(n, mean.shape[0])
        self.Y = np.array([
            0 if np.all(x < quadrant_bound) else 1
            for x in self.X
        ])

        for i in range(n):
            if random() < noise:
                self.Y[i] = 1 - self.Y[i]

        self.dim = mean.shape[0]
