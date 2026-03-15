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

    DEP forward pass equation:

    \\[ y = f(\\lambda \\tau_(x) + (1 - \\lambda) \\tau'_(x)) \\]

    Where $\\tau$ refers to the activation of a (max, +) morphological
    perceptron and $\\tau'$ to a (min, +) one.
    $\\lambda$ must be a number between 0 and 1.

    Fitting can be done by setting the constructor parameter 'method' to either:
    - dccp:  Use Disciplined Programming and the Convex-Concave Procedure.
             Compared to gradient descent, DCCP seems to converge faster.
    - wdccp: Use the same method, except apply weights to the cost
             contribution of each sample point, to lessen the impact of
             outliers in the training data.
             This is the default method, which seems to be more accurate in
             non-degenerate datasets.
    """

    def __init__(self, _lambda: float,
                 method: Literal['wdccp', 'dccp'] = 'wdccp',
                 max_iterations: int = 100, done_threshold: float = 1e-6,
                 verbose: bool = False) -> None:
        """
        Initialize the classifier, see class help for more.

        param _lambda:          lambda parameter for the DEP, see class help
        param method:           Either 'dccp' or 'wcddp'
        param max_iterations:   Upper bound for the number of iterations to use
                                during fitting
        param done_threshold:   The cost delta at which training is considered
                                finished
        param verbose:          Whether to log extra information
        """

        self._lambda = _lambda
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
        - self.fit_cost_:       Cached cost, fore use later by the user

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

        # create and train perceptrons
        N = X[0].shape[0]
        weighted = self.method == 'wdccp'
        trainer = DepDccpTrainer(N, self._lambda, weighted, self.max_iterations,
                                 self.done_threshold, self.verbose)

        self.fit_cost_ = trainer.train(X, y_integers)
        self.max_perceptron_ = trainer.max_perceptron
        self.min_perceptron_ = trainer.min_perceptron
        self._lambda = trainer._lambda

        if self.verbose:
            print(f'Cost after fit(): {self.fit_cost_:.2f}')

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return np.array([
            self.classes_[int(
                (self._lambda * self.max_perceptron_.forward(x) +
                 (1 - self._lambda) * self.min_perceptron_.forward(x)) >= 0)]
                for x in X])

    def get_decision_region_points(self
                                   ) -> tuple[tuple[np.ndarray, np.ndarray],
                                              tuple[np.ndarray, np.ndarray]]:
        """
        Return points and vectors defining the decision region of the DEP.
        The decision region can be described using the following diagram:

        +------------+
        |C_1     /v  |
        |   A___/    |
        |   /   B    |
        | u/      C_0|
        +------------+

        C_0 and C_1 refer to the classes, although they may not be in this exact
        configuration.
        This function returns [[A, u], [B, v]], where A and B are the points at
        which the decision region makes an angle and u/v are vectors for the
        external semilines [A) and [B) respectively.

        In the cases where lambda is either 0 or 1, only one point exists and
        the other goes towards infinity.
        In this case, A and B will refer to the same point and either u or v
        will refer to the segment from the existing point and the one at
        infinity, for an easier use.

        For example:

        +------------+
        |C_1     /v  |
        |_______/    |
        | <-u   B    |
        |         C_0|
        +------------+
        """

        # variable shortcuts for readability
        l = self._lambda
        w_max = self.max_perceptron_.weights
        w_min = self.min_perceptron_.weights
        N = self.max_perceptron_.dim

        # calculate the coordinates for the line between A and B
        axis = np.argmin(w_min - w_max)
        ab_constant_coord = -l * w_max[axis] - (1 - l) * w_min[axis]
        # alternative unchanging coords for later, using +/-1 this way because
        # the regions are always lower for max and upper for min
        alternative_coord = [ab_constant_coord - 1, ab_constant_coord + 1]

        # Calculate the coordinates of A and B, knowing they have one in common
        # being ab_constant_coord.
        # Also compute A_ and B_, points also on these lines, to compute u and v
        A, A_ = np.empty(N), np.empty(N)
        B, B_ = np.empty(N), np.empty(N)
        A[axis] = B[axis] = ab_constant_coord
        A_[axis], B_[axis] = alternative_coord[0], alternative_coord[1]

        for ax in range(N):
            if ax == axis:
                continue
            A[ax] = -w_max[ax] - (1-l)/l * (ab_constant_coord + w_min[axis])
            B[ax] = -w_min[ax] - l/(1-l) * (ab_constant_coord + w_max[axis])
            A_[ax] = -w_max[ax] - (1-l)/l * (alternative_coord[0] + w_min[axis])
            B_[ax] = -w_min[ax] - l/(1-l) * (alternative_coord[1] + w_max[axis])

        u = A_ - A
        v = B_ - B

        return ((A, u), (B, v))

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
