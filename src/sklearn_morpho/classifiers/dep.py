from __future__ import annotations

from typing import Literal, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Tags, check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from ..inversion import CentroidInversion, InversionHeuristic
from ..stopping import (
    CostStoppingMethod,
    EarlyStoppingMethod,
    EpochStoppingMethod,
    StoppingMethod,
    TrainStopStoppingMethod,
)
from ..training.dccp_dep import DEPDccpTrainer
from ..weighting import NoneSampleWeighting, SampleWeighting


class DEP(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn estimator wrapper around a DEP (Dilation-Erosion morphological
    Perceptron) for binary data classification.

    The DEP's activation function is defined as:

    \\[ y = f(\\lambda \\tau_(x) + (1 - \\lambda) \\tau'_(x)) \\]

    Where $\\tau$ refers to the activation of a (max, +) morphological
    perceptron and $\\tau'$ to a (min, +) one.
    $\\lambda$ is a real number between 0 and 1 to guarantee correct convexity,
    but in practice a smaller interval can be enforced to avoid imprecisions.
    """

    def __init__(
        self,
        lambda_bounds: tuple[float, float] = (1e-3, 1 - 1e-3),
        margin: float = 0.0,
        penalty: float = 0.0,
        validation_ratio: float = 0.3,
        weighting_method: SampleWeighting | None = None,
        stopping_methods: list[StoppingMethod] | None = None,
        inversion_method: InversionHeuristic | None = None,
        solver: str | None = None,
        verbose: Literal[0, 1, 2] = 0,
        random_state: np.random.RandomState | None = None,
    ) -> None:
        """
        Initialize the classifier, see class help for more.

        param lambda_bounds:    A pair of min and max values for lambda, to
                                avoid solvers (especially dccp) from failing to
                                optimize.
                                To keep the constraints at the right convexity,
                                the bounds must be inside [0, 1].
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
        param inversion_method: The heuristic to use to know whether to invert
                                the target classes, as the dataset's orientation
                                might not always be favorable.
                                If left to None, will use a CentroidInversion
                                optimizing for (1, 1, ..., 1).
        param solver:           The solver to use in cvxpy optimization.
                                If set to "dccp", will use the solver from the
                                dccp library instead of the customized DCA.
        param verbose:          Whether to log extra information. 0: no logging,
                                1: basic logging / timing, 2: cvxpy solve() set
                                to verbose mode.
        param random_state:     A RandomState object or None to allow for seeded
                                randomness.
        """

        self.lambda_bounds = lambda_bounds
        self.margin = margin
        self.penalty = penalty
        self.validation_ratio = validation_ratio
        self.weighting_method = weighting_method
        self.stopping_methods = stopping_methods
        self.inversion_method = inversion_method
        self.solver = solver
        self.verbose: Literal[0, 1, 2] = verbose
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> DEP:
        """
        Fit the classifier, create attributes:
        - self.max_perceptron_
        - self.min_perceptron_
        - self.classes_:        Unique labels generated from y

        X and y must represent binary classifiable data.
        """

        # input data validation
        random_state = check_random_state(self.random_state)
        X, y = validate_data(self, X, y)  # type: ignore
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
        if self.inversion_method is None:
            inversion_method: InversionHeuristic = CentroidInversion(
                np.ones(X.shape[1])
            )
        else:
            inversion_method = self.inversion_method

        # create classes and convert them to distinct integers for fitting
        # the classes are persisted inside the object for use in predict
        self.classes_ = unique_labels(y)
        classes_list = list(self.classes_)
        y_integers = np.array(
            [classes_list.index(c) for c in y], dtype=np.int32
        )

        if len(classes_list) != 2:
            raise ValueError(
                'Only binary classification is supported but '
                f'got {len(classes_list)} class(es).'
            )

        # create and train perceptrons
        trainer = DEPDccpTrainer(
            self.lambda_bounds,
            self.margin,
            self.penalty,
            self.validation_ratio,
            weighting_method,
            stopping_methods,
            inversion_method,
            self.solver,
            self.verbose,
            random_state,
        )

        trainer.train(X_scaled, y_integers)
        self.max_perceptron_ = trainer.max_perceptron
        self.min_perceptron_ = trainer.min_perceptron
        self.lambda_ = trainer.lambda_
        self.invert_res_ = trainer.invert_res

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)  # type: ignore
        X_scaled = cast(np.ndarray, self.scaler_.transform(X))

        expr_max = np.max(self.max_perceptron_ + X_scaled, axis=1)
        expr_min = np.min(self.min_perceptron_ + X_scaled, axis=1)
        activation = expr_max * self.lambda_ + expr_min * (1 - self.lambda_)

        res: np.ndarray = activation * (1 - 2 * self.invert_res_)
        return res

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        res: np.ndarray = self.classes_[
            (self.decision_function(X) >= 0).astype(int)
        ]
        return res

    def __sklearn_tags__(self) -> Tags:
        """
        Overriden method to allow check_estimator to not run accuracy tests.
        These are designed for perceptrons with a linear decision boundary,
        which is not the case for a morphological perceptron.
        """

        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
