from __future__ import annotations

import numpy as np
from typing import Literal, cast

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from ..dccp.dccp_simple_perceptron import SimplePerceptronDccpTrainer
from ..stopping import (
        StoppingMethod,
        CostStoppingMethod,
        EarlyStoppingMethod,
        EpochStoppingMethod,
        TrainStopStoppingMethod,
)
from ..weighting import SampleWeighting, NoneSampleWeighting

class MorphoPerceptron(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn estimator wrapper around a simple morphological perceptron.

    The morphological perceptron's activation function is defined as:

    \\[ y = \\tau(w + x) \\]

    Where $\\tau$ refers to max (resp. min), for a dilation (resp. erosion)
    perceptron and $w$ the perceptron's weights.
    """

    def __init__(self, kind: Literal['max', 'min'], margin = 0.,
                 penalty = 0., validation_ratio = .3,
                 weighting_method: SampleWeighting | None = None,
                 stopping_methods: list[StoppingMethod] | None = None,
                 use_dccp_library: bool = False, verbose: Literal[0, 1, 2] = 0,
                 random_state: np.random.RandomState | None = None) -> None:
        """
        Initialize the classifier, see class help for more.

        param kind:             Whether the perceptron is dilation or erosion.
        param margin:           Enforce a margin between the decision boundary
                                and the data. May help with linearly separable
                                datasets, but generally lower is more accurate.
        param penalty:          A penalty to add to the weights squared and
                                avoid them exploding.
                                Must be a small positive number like 1e-6, or
                                zero to disable penalty calculation altogether.
        param validation_radio: How much of the training set to dedicate to use
                                as validation during fitting.
                                Must be between 0 and 1 (inclusive, exclusive),
                                if set to exactly 0 then incompatible stopping
                                methods cannot be used (e.g. early stopping).
                                Ignored when using the dccp library solver.
        param weighting_method: The weighting method to use: apply weights to
                                the cost contribution of each data point to help
                                avoid outliers.
                                If left to None, will use NoneWeightingMethod().
        param stopping_methods: A list of stopping methods, must not be empty.
                                At each epoch, these methods will be
                                sequentially asked whether the training should
                                stop. In this case, epoch ends by rolling back
                                to the epoch with the best validation cost.
                                If left to None, will use:
                                [
                                    CostStoppingMethod(),
                                    EarlyStoppingMethod(),
                                    EpochStoppingMethod(),
                                    TrainStopStoppingMethod(),
                                ]
                                Ignored when using the dccp library solver.
        param use_dccp_library: Whether to use the dccp library as a solver or
                                a manual constraints linearization in fit.
        param verbose:          Whether to log extra information. 0: no logging,
                                1: basic logging / timing, 2: cvxpy solve() set
                                to verbose mode.
        param random_state:     A RandomState object or None to allow for seeded
                                randomness.
        """

        self.kind: Literal['max', 'min'] = kind
        self.margin = margin
        self.penalty = penalty
        self.validation_ratio = validation_ratio
        self.weighting_method = weighting_method
        self.stopping_methods = stopping_methods
        self.use_dccp_library = use_dccp_library
        self.verbose: Literal[0, 1, 2] = verbose
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> MorphoPerceptron:
        """
        Fit the classifier, create attributes:
        - self.weights
        - self.classes_:        Unique labels generated from y

        X and y must represent binary classifiable data.
        """

        # input data validation
        random_state = check_random_state(self.random_state)
        X, y = validate_data(self, X, y) # type: ignore
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # set unset parameters to their default values
        if self.weighting_method is None:
            weighting_method = NoneSampleWeighting()
        else:
            weighting_method = self.weighting_method
        if self.stopping_methods is None:
            stopping_methods = [
                CostStoppingMethod(),
                EarlyStoppingMethod(),
                EpochStoppingMethod(),
                TrainStopStoppingMethod(),
            ]
        else:
            stopping_methods = self.stopping_methods

        # create classes and convert them to distinct integers for fitting
        # the classes are persisted inside the object for use in predict
        self.classes_ = unique_labels(y)
        classes_list = list(self.classes_)
        y_integers = np.array([classes_list.index(c) for c in y],
                           dtype=np.int32)

        if len(classes_list) != 2:
            raise ValueError('Only binary classification is supported but '
                             f'got {len(classes_list)} class(es).')

        # create and train perceptrons
        trainer = SimplePerceptronDccpTrainer(
            self.kind, self.margin, self.penalty, self.validation_ratio,
            weighting_method, stopping_methods, self.use_dccp_library,
            self.verbose, random_state
        )

        trainer.train(X_scaled, y_integers)
        self.weights_ = trainer.weights

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False) # type: ignore
        X_scaled = cast(np.ndarray, self.scaler_.transform(X))

        expr = self.weights_ + X_scaled

        if self.kind == 'max':
            return expr.max(axis=1)
        return expr.min(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]

    def __sklearn_tags__(self):
        """
        Overriden method to allow check_estimator to not run accuracy tests.
        These are designed for perceptrons with a linear decision boundary,
        which is not the case for a morphological perceptron.
        """

        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags
