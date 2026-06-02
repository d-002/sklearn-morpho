from __future__ import annotations

import numpy as np

from . import SampleWeighting

class NoneSampleWeighting(SampleWeighting):
    """
    Weighting method that does not weight its inputs.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoneSampleWeighting:
        self.weights_ = np.ones(len(X))
        self.cost_normalizer_ = 1

        return self
