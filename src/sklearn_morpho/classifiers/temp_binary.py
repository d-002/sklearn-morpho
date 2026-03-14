import numpy as np
from typing import Literal

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from ..perceptron import MaxPerceptron
from ..training import train_dccp, train_gradient

class TempBinaryClassifier(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn estimator wrapper around a MaxPerceptron for binary data
    classification.
    The perceptron being max- means data can only be classified if the region 0
    sits in the lower orthant of the space, meaning it is also axis-aligned.
    Can fit the underlying perceptron using one of two methods:
    - dccp:     Use Disciplined Programming and the Convex-Concave Procedure.
                Compared to gradient descent, DCCP seems to converge faster.
    - wdccp:    Use the same method, except apply weights to the cost
                contribution of each sample point, so that outliers steer the
                decision region less to improve accuracy on non-separable
                datasets.
                This is the default method.
    - gradient: Use classic gradient descent method.
                This fitting method can be applied to more scenarios, but is
                slower and only included here for demonstration purposes.
    """

    def __init__(self, method: Literal['wdccp', 'dccp', 'gradient'] = 'wdccp',
                 n_iterations: int = 100,
                 done_threshold: float = 1e-6, verbose: bool = False):

        self.method = method
        self.n_iterations = n_iterations
        self.done_threshold = done_threshold
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> TempBinaryClassifier:
        # input data validation
        X, y = validate_data(self, X, y)

        # create classes and convert them to distinct integers for fitting
        # the classes are persisted inside the object for use in predict
        self.classes_ = unique_labels(y)
        classes_list = list(self.classes_)
        y_int = np.array([classes_list.index(c) for c in y],
                           dtype=np.int32)

        if len(classes_list) != 2:
            raise ValueError('Only binary classification is supported but '
                             f'got {len(classes_list)} class(es).')

        # create the MaxPerceptron this class is a wrapper of
        self.perceptron_ = MaxPerceptron(X[0].size)

        match self.method:
            case 'wdccp':
                self.last_cost_ = train_dccp(self.perceptron_, X, y_int,
                                             True, self.n_iterations,
                                             self.done_threshold, self.verbose)
            case 'dccp':
                self.last_cost_ = train_dccp(self.perceptron_, X, y_int,
                                             False, self.n_iterations,
                                             self.done_threshold, self.verbose)
            case 'gradient':
                self.last_cost_ = train_gradient(self.perceptron_, X, y_int,
                                                 self.n_iterations,
                                                 self.done_threshold,
                                                 verbose = self.verbose)

        if self.verbose:
            print(f'Cost {self.last_cost_:.2f}')

        return self

    def get_fit_cost(self) -> float:
        """
        Additional method used to retrieve the computed cost from the last call
        to fit.
        Lower is better, with 0 meaning the call to fit managed to create a
        perceptron able to correctly classify all training data.
        """

        return self.last_cost_

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return np.array([self.classes_[int(self.perceptron_.forward(x) >= 0)]
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
