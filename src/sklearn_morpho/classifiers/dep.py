import numpy as np
from typing import Literal

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from ..training.dccp_dep import DepDccpTrainer

class DilationErosionPerceptron(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn estimator wrapper around a DEP (Dilation-Erosion morphological
    Perceptron) for binary data classification.

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
                 max_iterations: int = 100, done_threshold: float = 1e-6,
                 verbose: bool = False) -> None:
        """
        Initialize the classifier, see help(self.__class__) for more.

        param method:           Either 'dccp' or 'wcddp'
        param max_iterations:   Upper bound for the number of iterations to use
                                during fitting
        param done_threshold:   The cost delta at which training is considered
                                finished
        param verbose:          Whether to log extra information
        """

        self.method = method
        self.max_iterations = max_iterations
        self.done_threshold = done_threshold
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> DilationErosionPerceptron:
        """
        Fit the classifier, create attributes:
        - self.max_perceptron_
        - self.min_perceptron_
        - self.lambda
        - self.classes_:        Unique labels generated from y

        X and y must represent binary classifiable data.
        """

        # input data validation
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

        # create or override perceptrons
        N = X[0].shape[0]

        # train perceptrons
        trainer = DepDccpTrainer(N, self.method == 'wdccp', self.max_iterations,
                                 self.done_threshold, self.verbose)
        cost = trainer.train(X, y_integers)
        self.max_perceptron_ = trainer.max_perceptron
        self.min_perceptron_ = trainer.min_perceptron
        self.lambda_ = trainer.lambda_

        if self.verbose:
            print(f'Cost after fit(): {cost:.2f}')

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return np.array([
            self.classes_[
                (self.lambda_ * self.max_perceptron_.forward(x)) +
                ((1 - self.lambda_) * self.min_perceptron_.forward(x))]
            for x in X])

    def __sklearn_tags__(self):
        """
        Overriden method to allow check_estimator to not run accuracy tests.
        These are designed for perceptrons with a linear decision boundary,
        which is not the case for a morphological perceptron.
        """

        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        tags.classifier_tags.multi_class = False
        return tags
