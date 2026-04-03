from typing import Any, Literal
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

        super().__init__([self.max_perceptron, self.min_perceptron], weighted,
                         margin, max_iterations, batch_size, done_threshold,
                         verbose, rs)

    def at_training_start(self) -> None:
        super().at_training_start()

        self._objective = None

        # Create transformation matrices constraints, as well as original values
        # for linearization.
        # Use these to absorbe the final lambda parameter, restored at the end,
        # inspired by arXiv:2011.06512v1.
        N = self.max_perceptron.dim
        self._max_training_matrix = cp.Variable((N, N))
        self._min_training_matrix = cp.Variable((N, N))
        self.max_matrix = self.rs.rand(N, N)
        self.min_matrix = self.rs.rand(N, N)

    def get_problem(self, weights: list[cp.Variable], X: np.ndarray,
                    Y: np.ndarray, wdccp_weights: np.ndarray) -> tuple[
                            cp.Minimize | cp.Maximize, list[cp.Constraint]]:
        K = X.shape[0]

        # the objective and the slack variables do not change, cache them
        if self._objective is None:
            self._slack = cp.Variable(K)
            self._objective = cp.Minimize(
                    cp.sum(cp.multiply(cp.pos(self._slack), wdccp_weights)))

        # Constraints: convex constraints are for data points in the first
        # class, while concave ones are for points in the second class.
        # Therefore, linearize things when needed
        max_weights, min_weights = weights
        constraints = [None] * K
        for i, (x, y) in enumerate(zip(X, Y)):
            if y == 0:
                index = np.argmin(min_weights + self.min_matrix @ x)
                value = cp.max(max_weights + self._max_training_matrix @ x) + \
                        (min_weights + self._min_training_matrix @ x)[index]
                constraints[i] = self._slack[i] >= self.margin + value
            else:
                index = np.argmax(max_weights + self.max_matrix @ x)
                value = (max_weights + self._max_training_matrix @ x)[index] + \
                        cp.min(min_weights + self._min_training_matrix @ x)
                constraints[i] = self._slack[i] >= self.margin - value

        return self._objective, constraints

    def after_iteration(self) -> None:
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
