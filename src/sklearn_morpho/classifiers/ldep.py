import numpy as np
from typing import Literal, cast

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from ..dccp.dccp_ldep import LDEPDccpTrainer
from ..stopping import (
        StoppingMethod,
        CostStoppingMethod,
        EarlyStoppingMethod,
        EpochStoppingMethod,
        TrainStopStoppingMethod,
)
from ..weighting import SampleWeighting, NoneSampleWeighting

class LDEP(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn estimator wrapper around a l-DEP (linear Dilation-Erosion
    morphological Perceptron) for binary data classification.

    l-DEP forward pass equation:

    \\[ y = f(\\lambda \\tau_(R_1(x)) + (1 - \\lambda) \\tau'_(R_2(x))) \\]

    Where $\\tau$ refers to the activation of a (max, +) morphological
    perceptron and $\\tau'$ to a (min, +) one.
    $R_1, R_2$ are linear transformations to apply to the training data.
    They convert it into a latent space with possibly different dimensions,
    also allowing classification of arbitrarily distributed data.
    A higher dimension for the latent space will result in slower training
    times, but will allow the decision boundary to be more complex.
    """

    def __init__(self, latent_dims: tuple[int, int] = (10, 10), margin = 1.,
                 validation_ratio = .3,
                 weighting_method: SampleWeighting | None = None,
                 stopping_methods: list[StoppingMethod] | None = None,
                 verbose: Literal[0, 1, 2] = 0,
                 solver: Literal['dccp'] | None = None,
                 random_state: np.random.RandomState | None = None) -> None:
        """
        Initialize the classifier, see class help for more.

        param latent_dims:      The dimensions of the latent spaces used for the
                                linear transformations output
        param margin:           Enforce a margin between the decision boundary
                                and the data. May help with linearly separable
                                datasets, but generally lower is more accurate.
        param validation_radio: How much of the training set to dedicate to use
                                as validation during fitting.
                                Must be between 0 and 1 (inclusive, exclusive),
                                if set to exactly 0 then incompatible stopping
                                methods cannot be used (e.g. early stopping).
        param weighting_method: The weighting method to use: apply weights to
                                the cost contribution of each data point to help
                                avoid outliers.
                                If left to None, will use NoneWeightingMethod()
        param stopping_methods: A list of stopping methods, must not be empty.
                                At each epoch, these methods will be
                                sequentially asked whether the training should
                                stop. In this case, epoch ends by rolling back
                                to the epoch with the best validation cost.
                                If left to None, will use:
                                [
                                    CostStoppingMethod(1e-6),
                                    EarlyStoppingMethod(5),
                                    EpochStoppingMethod(20),
                                    TrainStopStoppingMethod(),
                                ]
        param verbose:          Whether to log extra information. 0: no logging,
                                1: basic logging / timing, 2: cvxpy solve() set
                                to verbose mode.
        param solver:           The solver to use in cvxpy optimizations, or
                                None for default.
        param random_state:     A RandomState object or None to allow for seeded
                                randomness.
        """

        self.latent_dims = latent_dims
        self.margin = margin
        self.validation_ratio = validation_ratio
        self.weighting_method = weighting_method
        self.stopping_methods = stopping_methods
        self.verbose: Literal[0, 1, 2] = verbose
        self.solver: Literal['dccp'] | None = solver
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> LDEP:
        """
        Fit the classifier, create attributes:
        - self.max_perceptron_
        - self.min_perceptron_
        - self.lambda_
        - self.max_matrix_
        - self.min_matrix_
        - self.classes_:        Unique labels generated from y

        X and y must represent binary classifiable data.
        """

        # input data validation
        random_state = check_random_state(self.random_state)
        X, y = validate_data(self, X, y)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # set unset parameters to their default values
        if self.weighting_method is None:
            weighting_method = NoneSampleWeighting()
        else:
            weighting_method = self.weighting_method
        if self.stopping_methods is None:
            stopping_methods = [
                CostStoppingMethod(1e-6),
                EarlyStoppingMethod(5),
                EpochStoppingMethod(20),
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
        trainer = LDEPDccpTrainer(self.latent_dims, self.margin,
                                  self.validation_ratio, weighting_method,
                                  stopping_methods, self.solver, self.verbose,
                                  random_state)

        trainer.train(X_scaled, y_integers)
        self.max_perceptron_ = trainer.max_perceptron
        self.min_perceptron_ = trainer.min_perceptron
        self.lambda_ = trainer.lambda_
        self.max_matrix_ = trainer.max_matrix
        self.min_matrix_ = trainer.min_matrix

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X_scaled = cast(np.ndarray, self.scaler_.transform(X))

        expr_max = np.max(self.max_perceptron_ + X_scaled @ self.max_matrix_.T,
                          axis=1)
        expr_min = np.min(self.min_perceptron_ + X_scaled @ self.min_matrix_.T,
                          axis=1)
        return expr_max * self.lambda_ + expr_min * (1 - self.lambda_)

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
        return tags
