import numpy as np
from typing import Literal

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from ..training.dccp_ldep import LDEPDccpTrainer
from ..weighting.weighting_base import SampleWeighting
from ..weighting.not_weighted import NoSampleWeighting

class LDEP(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn estimator wrapper around a l-DEP (linear Dilation-Erosion
    morphological Perceptron) for binary data classification.

    DEP forward pass equation:

    \\[ y = f(\\lambda \\tau_(R_1(x)) + (1 - \\lambda) \\tau'_(R_2(x))) \\]

    Where $\\tau$ refers to the activation of a (max, +) morphological
    perceptron and $\\tau'$ to a (min, +) one.
    $R_1, R_2$ are linear transformations to apply to the training data.
    They convert it into a latent space with possibly different dimensions,
    also allowing classification of arbitrarily distributed data.
    A higher dimension for the latent space will result in slower training
    times, but will allow the decision boundary to be more complex.
    """

    def __init__(self, weighting_method: SampleWeighting | None = None,
                 latent_dims: tuple[int, int] = (10, 10),
                 margin = 0., max_iterations = 100, batch_size = 32,
                 done_threshold = 1e-9, verbose: Literal[0, 1, 2] = 0,
                 random_state: np.random.RandomState | None = None) -> None:
        """
        Initialize the classifier, see class help for more.

        param latent_dims:      The dimensions of the latent spaces used for the
                                linear transformations output
        param weighting_method: The weighting method to use: apply weights to
                                the cost contribution of each data point to help
                                avoid outliers.
        param margin:           Enforce a margin between the decision boundary
                                and the data. May help with linearly separable
                                datasets, but generally lower is more accurate.
        param max_iterations:   Upper bound for the number of iterations to use
                                during fitting.
        param batch_size:       For mini batch fitting, define the batch size.
                                If zero, don't use batching.
        param done_threshold:   The rate of change for the cost between
                                consecutive iterations under which training is
                                considered finished.
        param verbose:          Whether to log extra information. 0: no logging,
                                1: basic logging / timing, 2: cvxpy solve() set
                                to verbose mode.
        param random_state:     A RandomState object or None to allow for seeded
                                randomness.
        """

        self.latent_dims = latent_dims
        self.weighting_method = weighting_method
        self.margin = margin
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.done_threshold = done_threshold
        self.verbose: Literal[0, 1, 2] = verbose
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> LDEP:
        """
        Fit the classifier, create attributes:
        - self.max_perceptron_
        - self.min_perceptron_
        - self.lambda
        - self.classes_:        Unique labels generated from y
        - self.fit_cost_:       Cached cost, fore use later by the user

        X and y must represent binary classifiable data.
        """

        # input data validation
        random_state = check_random_state(self.random_state)
        X, y = validate_data(self, X, y)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

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
        if self.weighting_method is None:
            weighting_method = NoSampleWeighting()
        else:
            weighting_method = self.weighting_method
        trainer = LDEPDccpTrainer(X_scaled.shape[1], self.latent_dims,
                                  weighting_method, self.margin,
                                  self.max_iterations, self.batch_size,
                                  self.done_threshold, self.verbose,
                                  random_state)

        self.fit_cost_ = trainer.train(X_scaled, y_integers)
        self.max_perceptron_ = trainer.max_perceptron
        self.min_perceptron_ = trainer.min_perceptron
        self.lambda_ = trainer.lambda_
        self.max_matrix_ = trainer.max_matrix
        self.min_matrix_ = trainer.min_matrix

        if self.verbose:
            print(f'Cost after fit(): {self.fit_cost_:.8f}')

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X_scaled = self.scaler_.transform(X)

        a, b = self.lambda_, 1 - self.lambda_
        return np.array([
            a * self.max_perceptron_.forward(self.max_matrix_ @ x) +
            b * self.min_perceptron_.forward(self.min_matrix_ @ x)
            for x in X_scaled])

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
