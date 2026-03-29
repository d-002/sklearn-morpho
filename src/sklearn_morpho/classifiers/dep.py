import numpy as np
from typing import Literal

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from ..training.dccp_dep import DepDccpTrainer

class DilationErosionPerceptron(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn estimator wrapper around a l-DEP (linear Dilation-Erosion
    morphological Perceptron) for binary data classification.

    DEP forward pass equation:

    \\[ y = f(\\lambda \\tau_(\\rho(x)) + (1 - \\lambda) \\tau'_(\\rho(x))) \\]

    Where $\\tau$ refers to the activation of a (max, +) morphological
    perceptron and $\\tau'$ to a (min, +) one.
    $\\rho$ is a linear transformation to apply to the training data to be able
    to classify any distribution of binary data.

    Fitting can be done by setting the constructor parameter 'method' to either:
    - dccp:  Use Disciplined Programming and the Convex-Concave Procedure.
             Compared to gradient descent, DCCP seems to converge faster.
    - wdccp: Use the same method, except apply weights to the cost
             contribution of each sample point, to lessen the impact of
             outliers in the training data.
             This is the default method, which seems to be more accurate in
             non-degenerate datasets.
    """

    def __init__(self, method: Literal['wdccp', 'dccp'] = 'wdccp',
                 margin: float = 0, max_iterations: int = 100,
                 done_threshold: float = 1e-9, verbose: bool = False,
                 random_state: np.random.RandomState | None = None) -> None:
        """
        Initialize the classifier, see class help for more.

        param method:         Either 'dccp' or 'wcddp'
        param margin:         Enforce a margin between the decision boundary and
                              the data. May help with linearly separable
                              datasets, but generally lower is more accurate.
        param max_iterations: Upper bound for the number of iterations to use
                              during fitting.
        param done_threshold: The rate of change for the cost between
                              consecutive iterations under which training is
                              considered finished.
        param verbose:        Whether to log extra information / time fit().
        param random_state:   A RandomState object for predictable randomness,
                              or None
        """

        self.method = method
        self.margin = margin
        self.max_iterations = max_iterations
        self.done_threshold = done_threshold
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> DilationErosionPerceptron:
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
        rs = check_random_state(self.random_state)
        X, y = validate_data(self, X, y)

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
        N = X[0].shape[0]
        weighted = self.method == 'wdccp'
        trainer = DepDccpTrainer(N, weighted, self.margin, self.max_iterations,
                                 self.done_threshold, self.verbose, rs)

        self.fit_cost_ = trainer.train(X, y_integers)
        self.max_perceptron_ = trainer.max_perceptron
        self.min_perceptron_ = trainer.min_perceptron
        self.lambda_ = trainer.lambda_
        self.max_matrix_ = trainer.max_matrix
        self.min_matrix_ = trainer.min_matrix

        if self.verbose:
            print(f'Cost after fit(): {self.fit_cost_:.2f}')

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        a, b = self.lambda_, 1 - self.lambda_
        return np.array([
            a * self.max_perceptron_.forward(self.max_matrix_ @ x) +
            b * self.min_perceptron_.forward(self.min_matrix_ @ x)
            for x in X])

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
