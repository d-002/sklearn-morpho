from typing import Literal, cast
import numpy as np
import cvxpy as cp

from .dccp_wrapper import DccpTrainer
from ..perceptron import MaxPerceptron, MinPerceptron

class DepDccpTrainer(DccpTrainer):
    def __init__(self, data_dim: int, latent_dims: tuple[int, int],
                 weighted: bool, margin: float, max_iterations: int,
                 batch_size: int, done_threshold: float,
                 verbose: Literal[0, 1, 2],
                 random_state: np.random.RandomState) -> None:
        """
        Initialize the dilation-erosion perceptron trainer.

        param data_dim:    the dimensionality of the input data
        param latent_dims: the latent dimensions for the max and min perceptrons
        param [others]:    see base class
        """

        self.data_dim = data_dim
        self.max_perceptron = MaxPerceptron(latent_dims[0])
        self.min_perceptron = MinPerceptron(latent_dims[1])

        super().__init__(weighted, margin, max_iterations, batch_size,
                         done_threshold, verbose, random_state)

    def at_training_start(self) -> None:
        N_max, N_min = self.max_perceptron.dim, self.min_perceptron.dim
        N_data = self.data_dim

        self._objective = None

        # Extracted parameters that will be populated during training but need
        # initial values for linearization
        for perceptron in (self.max_perceptron, self.min_perceptron):
            perceptron.weights = \
                    self.random_state.random(perceptron.dim) * 2 - 1
        self.max_matrix = self.random_state.rand(N_max, N_data)
        self.min_matrix = self.random_state.rand(N_min, N_data)

        # Create constraints for linearization derived from real parameters,
        # used to absorbe non-convex parameters that are restored at the end.
        # Method inspired by arXiv:2011.06512v1.
        self._max_training_weights = cp.Variable(N_max)
        self._min_training_weights = cp.Variable(N_min)
        self._max_training_matrix = cp.Variable((N_max, N_data))
        self._min_training_matrix = cp.Variable((N_min, N_data))

    def get_problem(self, X: np.ndarray, y: np.ndarray,
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

        idx_max = np.argmax(self.max_perceptron.weights + X @ self.max_matrix.T,
                            axis=1)
        idx_min = np.argmin(self.min_perceptron.weights + X @ self.min_matrix.T,
                            axis=1)

        constraints = [None] * K
        for i in range(K):
            expr_max = self._max_training_weights \
                    + self._max_training_matrix @ X[i]
            expr_min = self._min_training_weights \
                    + self._min_training_matrix @ X[i]
            if y[i] == 0:
                value = cp.max(expr_max) + expr_min[idx_min[i]]
                constraints[i] = self._slack[i] >= self.margin + value
            else:
                value = expr_max[idx_max[i]] + cp.min(expr_min)
                constraints[i] = self._slack[i] >= self.margin - value
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
