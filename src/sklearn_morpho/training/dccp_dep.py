from typing import Literal, cast
import numpy as np
import cvxpy as cp

from .dccp_wrapper import DccpTrainer
from ..perceptron import MaxPerceptron, MinPerceptron

class DepDccpTrainer(DccpTrainer):
    def __init__(self, N: int, weighted: bool, margin: float,
                 max_iterations: int, batch_size: int, done_threshold: float,
                 verbose: Literal[0, 1, 2], rs: np.random.RandomState) -> None:
        """
        Initialize the dilation-erosion perceptron trainer.

        param N:        The dimension of the data.
        param [others]: See base class.
        """

        self.max_perceptron = MaxPerceptron(N)
        self.min_perceptron = MinPerceptron(N)

        super().__init__(weighted, margin, max_iterations, batch_size,
                         done_threshold, verbose, rs)

    def at_training_start(self) -> None:
        N = self.max_perceptron.dim
        self._objective = None

        # Extracted parameters that will be populated during training but need
        # initial values for linearization
        for perceptron in (self.max_perceptron, self.min_perceptron):
            perceptron.weights = np.random.random(perceptron.dim) * 2 - 1
        self.max_matrix = self.rs.rand(N, N)
        self.min_matrix = self.rs.rand(N, N)

        # Create constraints for linearization derived from real parameters,
        # used to absorbe non-convex parameters that are restored at the end.
        # Method inspired by arXiv:2011.06512v1.
        self._max_training_weights = cp.Variable(N)
        self._min_training_weights = cp.Variable(N)
        self._max_training_matrix = cp.Variable((N, N))
        self._min_training_matrix = cp.Variable((N, N))

    def get_problem(self, X: np.ndarray, Y: np.ndarray,
                    wdccp_weights: np.ndarray
                    ) -> tuple[cp.Minimize | cp.Maximize, list[cp.Constraint]]:
        K = X.shape[0]

        # the objective and the slack variables do not change, cache them
        if self._objective is None:
            self._slack = cp.Variable(K)
            self._objective = cp.Minimize(
                    cp.sum(cp.multiply(cp.pos(self._slack), wdccp_weights)))

        # Constraints: convex constraints are for data points in the first
        # class, while concave ones are for points in the second class.
        # Therefore, linearize things when needed to make the problem convex,
        # sometimes using values from the previous iteration.
        index_max = np.argmax(self.max_perceptron.weights
                              + X @ self.max_matrix.T, axis=1)
        index_min = np.argmin(self.min_perceptron.weights
                              + X @ self.min_matrix.T, axis=1)

        expr_max = self._max_training_weights + X @ self._max_training_matrix.T
        expr_min = self._min_training_weights + X @ self._min_training_matrix.T

        # use masks and matrix multiplications/sums for efficiency, to avoid
        # using for loops or dimension mismatches
        mask_max = np.zeros(expr_max.shape)
        mask_min = np.zeros(expr_min.shape)
        mask_max[np.arange(K), index_max] = 1
        mask_min[np.arange(K), index_min] = 1

        lin_max = cast(cp.Variable, cp.sum(cp.multiply(expr_max, mask_max),
                                           axis=1))
        lin_min = cast(cp.Variable, cp.sum(cp.multiply(expr_min, mask_min),
                                           axis=1))

        cp_max = cp.max(expr_max, axis=1)
        cp_min = cp.min(expr_min, axis=1)

        constraints = []
        if np.any(Y == 0):
            value = cp_max[Y == 0] + lin_min[Y == 0]
            constraints.append(self._slack[Y == 0] >= self.margin + value)
        if np.any(Y == 1):
            value = cp_min[Y == 1] + lin_max[Y == 1]
            constraints.append(self._slack[Y == 1] >= self.margin - value)

        return self._objective, constraints

    def after_iteration(self) -> None:
        # update the perceptrons weights from this iteration's results
        for perceptron, w in zip((self.max_perceptron, self.min_perceptron),
                                 (self._max_training_weights,
                                  self._min_training_weights)):
            if w.value is None:
                raise ValueError('CvxPy could not optimize a perceptron')
            perceptron.weights = w.value

        # extract the matrices
        if self._max_training_matrix.value is None \
                or self._min_training_matrix.value is None:
            raise ValueError('CvxPy could not optimize transformation matrices')
        self.max_matrix = self._max_training_matrix.value
        self.min_matrix = self._min_training_matrix.value

    def at_training_end(self) -> None:
        max_matrix_norm = np.linalg.norm(self.max_matrix)
        min_matrix_norm = np.linalg.norm(self.min_matrix)

        div = max_matrix_norm + min_matrix_norm
        if not div:
            raise ValueError('Transformation matrices are zero, cannot resolve')

        self.lambda_ = max_matrix_norm / div
        self.max_matrix /= self.lambda_
        self.min_matrix /= 1 - self.lambda_
        self.max_perceptron.weights /= self.lambda_
        self.min_perceptron.weights /= 1 - self.lambda_
