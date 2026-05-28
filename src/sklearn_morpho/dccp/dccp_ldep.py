from typing import Literal
import numpy as np
import cvxpy as cp

from .dccp_wrapper import DccpTrainer
from ..weighting import SampleWeighting
from ..stopping import StoppingMethod

class LDEPDccpTrainer(DccpTrainer):
    def __init__(self, latent_dims: tuple[int, int],
                 margin: float, validation_ratio: float,
                 weighting_method: SampleWeighting,
                 stopping_methods: list[StoppingMethod],
                 solver: Literal['dccp'] | None,
                 verbose: Literal[0, 1, 2],
                 random_state: np.random.RandomState) -> None:
        """
        Initialize the l-DEP trainer.

        param latent_dims: the latent dimensions for the max and min perceptrons
        param [others]:    see base class
        """

        self.latent_dims = latent_dims

        super().__init__(margin, validation_ratio, weighting_method,
                         stopping_methods, solver, verbose, random_state)

    def at_training_start(self, data_dim: int) -> None:
        N_max, N_min = self.latent_dims

        self._objective = None

        # Extracted parameters that will be populated during training but need
        # initial values for linearization
        self.max_perceptron = self.random_state.randn(N_max)
        self.min_perceptron = self.random_state.randn(N_min)
        self.max_matrix = self.random_state.randn(N_max, data_dim)
        self.min_matrix = self.random_state.randn(N_min, data_dim)

        # Create constraints for linearization derived from real parameters,
        # used to absorbe non-convex parameters that are restored at the end.
        # Method inspired by arXiv:2011.06512v1.
        # TODO: is this representation an issue since the final lambda parameter
        # is not necessarily in [0, 1]?
        self._max_training_weights = cp.Variable(N_max)
        self._min_training_weights = cp.Variable(N_min)
        self._max_training_matrix = cp.Variable((N_max, data_dim))
        self._min_training_matrix = cp.Variable((N_min, data_dim))

    def get_problem_unproven(self, X: np.ndarray, y: np.ndarray,
                             cost_weights: np.ndarray) -> cp.Problem:
        K = X.shape[0]

        # the objective and the slack variables do not change, cache them
        if self._objective is None:
            self._slack = cp.Variable(K)
            self._objective = cp.Minimize(
                    cp.sum(cp.multiply(cp.pos(self._slack), cost_weights)))

        # Constraints: convex constraints are for data points in the first
        # class, while concave ones are for points in the second class.
        # Therefore, linearize things when needed to make the problem convex,
        # sometimes using values from the previous epoch.

        idx_max = np.argmax(self.max_perceptron + X @ self.max_matrix.T,
                            axis=1)
        idx_min = np.argmin(self.min_perceptron + X @ self.min_matrix.T,
                            axis=1)

        # create arrays to regroup the active indices using numpy, to then apply
        # constraints all at once and use AST optimizations inside cvxpy
        M_max = np.zeros((K, self.max_perceptron.size))
        M_min = np.zeros((K, self.min_perceptron.size))
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
                                          (1, self.max_perceptron.size),
                                          order='C')
            expr_min += ones @ cp.reshape(self._min_training_weights,
                                          (1, self.min_perceptron.size),
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

    def get_problem_dccp(self, X: np.ndarray, y: np.ndarray,
                         cost_weights: np.ndarray) -> cp.Problem:
        K = X.shape[0]

        if self._objective is None:
            self._slack = cp.Variable(K)
            self._objective = cp.Minimize(
                    cp.sum(cp.multiply(cp.pos(self._slack), cost_weights)))

        # Constraints: convex constraints are for data points in the first
        # class, while concave ones are for points in the second class.

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

            if label == 0:
                constraints.append(self._slack[mask] >= self.margin
                                   + cp.max(expr_max, axis=1) + active_min)
            else:
                constraints.append(self._slack[mask] >= self.margin
                                   - active_max - cp.min(expr_min, axis=1))
        return cp.Problem(self._objective, constraints)

    def get_problem(self, X: np.ndarray, y: np.ndarray,
                    cost_weights: np.ndarray) -> cp.Problem:
        if self.solver == 'dccp':
            return self.get_problem_dccp(X, y, cost_weights)
        return self.get_problem_unproven(X, y, cost_weights)

    def get_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        expr_max = np.max(self.max_perceptron + X @ self.max_matrix.T, axis=1)
        expr_min = np.min(self.min_perceptron + X @ self.min_matrix.T, axis=1)

        # can use the expressions directly without needing to multiply by lambda
        # since still in the training phase
        cost = self.margin + (expr_max + expr_min) * (1 - 2 * y)
        return np.maximum(0, cost).sum()

    def after_epoch(self) -> None:
        # update the perceptrons weights from this epoch's results
        if self._max_training_weights.value is None or \
                self._min_training_weights.value is None:
            raise ValueError('CvxPy could not optimize a perceptron')

        self.max_perceptron = self._max_training_weights.value
        self.min_perceptron = self._min_training_weights.value

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
        self.max_perceptron /= self.lambda_
        self.min_perceptron /= 1 - self.lambda_

    def save_best(self) -> None:
        self.saved = {
            'max_w': np.copy(self.max_perceptron),
            'min_w': np.copy(self.min_perceptron),
            'max_m': np.copy(self.max_matrix),
            'min_m': np.copy(self.min_matrix),
        }

    def rollback_to_best(self) -> None:
        self.max_perceptron = self.saved['max_w']
        self.min_perceptron = self.saved['min_w']
        self.max_matrix = self.saved['max_m']
        self.min_matrix = self.saved['min_m']
