from typing import Literal
import numpy as np
import cvxpy as cp

from .dccp_wrapper import DccpTrainer
from ..perceptron import MaxPerceptron, MinPerceptron
from ..weighting.weighting_base import SampleWeighting
from ..stopping.stopping_base import StoppingMethod

class LDEPDccpTrainer(DccpTrainer):
    def __init__(self, latent_dims: tuple[int, int],
                 margin: float, validation_ratio: float,
                 weighting_method: SampleWeighting,
                 stopping_methods: list[StoppingMethod],
                 verbose: Literal[0, 1, 2],
                 random_state: np.random.RandomState) -> None:
        """
        Initialize the dilation-erosion perceptron trainer.

        param latent_dims: the latent dimensions for the max and min perceptrons
        param [others]:    see base class
        """

        self.max_perceptron = MaxPerceptron(latent_dims[0])
        self.min_perceptron = MinPerceptron(latent_dims[1])

        super().__init__(margin, validation_ratio, weighting_method,
                         stopping_methods, verbose, random_state)

    def at_training_start(self, data_dim: int) -> None:
        N_max, N_min = self.max_perceptron.dim, self.min_perceptron.dim

        self._objective = None

        # Extracted parameters that will be populated during training but need
        # initial values for linearization
        for perceptron in (self.max_perceptron, self.min_perceptron):
            perceptron.weights = self.random_state.randn(perceptron.dim)
        self.max_matrix = self.random_state.randn(N_max, data_dim)
        self.min_matrix = self.random_state.randn(N_min, data_dim)

        # Create constraints for linearization derived from real parameters,
        # used to absorbe non-convex parameters that are restored at the end.
        # Method inspired by arXiv:2011.06512v1.
        self._max_training_weights = cp.Variable(N_max)
        self._min_training_weights = cp.Variable(N_min)
        self._max_training_matrix = cp.Variable((N_max, data_dim))
        self._min_training_matrix = cp.Variable((N_min, data_dim))

    def get_problem(self, X: np.ndarray, y: np.ndarray,
                    wdccp_weights: np.ndarray) -> cp.Problem:
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

        print(self.max_perceptron.weights.shape, X.shape, self.max_matrix.T.shape)
        idx_max = np.argmax(self.max_perceptron.weights + X @ self.max_matrix.T,
                            axis=1)
        idx_min = np.argmin(self.min_perceptron.weights + X @ self.min_matrix.T,
                            axis=1)

        # create encoding matrices for the active indices using numpy, to then
        # apply constraints all at once and use AST optimizations inside cvxpy
        M_max = np.zeros((K, self.max_perceptron.dim))
        M_min = np.zeros((K, self.min_perceptron.dim))
        M_max[np.arange(K), idx_max] = 1
        M_min[np.arange(K), idx_min] = 1

        constraints = []
        for label in [0, 1]:
            mask = y == label
            if not np.any(mask):
                continue

            X_ = X[mask]
            K_ = X_.shape[0]

            # start building the perceptron's outputs before the min/max
            expr_max = X_ @ self._max_training_matrix.T
            expr_min = X_ @ self._min_training_matrix.T

            # add weights to every row of (X @ matrix.T) using np.ones to create
            # a matrix safely for cvxpy's cpp backend
            ones = np.ones((K_, 1))
            expr_max += ones @ cp.reshape(self._max_training_weights,
                                          (1, self.max_perceptron.dim),
                                          order='C')
            expr_min += ones @ cp.reshape(self._min_training_weights,
                                          (1, self.min_perceptron.dim),
                                          order='C')

            active_max = cp.sum(cp.multiply(M_max[mask], expr_max), axis=1)
            active_min = cp.sum(cp.multiply(M_min[mask], expr_min), axis=1)

            if label == 0:
                constraints.append(self._slack[mask] >= self.margin
                                   + cp.max(expr_max, axis=1) + active_min)
            else:
                constraints.append(self._slack[mask] >= self.margin
                                   - active_max - cp.min(expr_min, axis=1))
        return cp.Problem(self._objective, constraints)

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
        if np.isclose(div, 0):
            raise ValueError('Transformation matrices are zero, cannot solve')

        self.lambda_ = max_matrix_norm / div
        self.max_matrix /= self.lambda_
        self.min_matrix /= 1 - self.lambda_
        self.max_perceptron.weights /= self.lambda_
        self.min_perceptron.weights /= 1 - self.lambda_
