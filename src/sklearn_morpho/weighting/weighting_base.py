import numpy as np
from abc import ABC

from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator

WeightingResult = tuple[np.ndarray, float]


class SampleWeighting(
    ABC, OneToOneFeatureMixin, TransformerMixin, BaseEstimator
):
    """
    Abstract class for transformers that convert data points to a set of
    respective weights.
    These weights may be used as the cost contribution in a DCCP problem.

    For example, a sample weighting with functionality `np.full(len(X), 1)`
    corresponds to no weighting at all.

    The fit() method must create both:
    - self.weights_: np.ndarray
    - self.cost_normalizer_: float

    If the implementation does not follow these guidelines, the weighting should
    still be able to provide a WeightingResult as the return value of
    transform().
    In this case, you are free to override the transform() and fit_transform()
    methods containing boilerplate code.
    """

    # for LSPs
    def __init__(self):
        self.weights_: np.ndarray
        self.cost_normalizer_: float

    def transform(self, X: np.ndarray | None) -> WeightingResult:
        if X is not None:
            raise ValueError(
                f'{self.__class__.__name__}: only supports '
                'fitting and transforming the same data, do not '
                'specify it in transform()'
            )

        return self.weights_, self.cost_normalizer_

    # override this to make fit_transform usable but still prevent wrong usage
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,  # pyright: ignore
    ) -> WeightingResult:
        self.fit(X, y)  # pyright: ignore
        return self.transform(None)
