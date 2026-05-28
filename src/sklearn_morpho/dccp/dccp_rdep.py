from typing import Literal
import numpy as np
import cvxpy as cp

from .dccp_wrapper import DccpTrainer
from ..weighting import SampleWeighting
from ..stopping import StoppingMethod

class RDEPDccpTrainer(DccpTrainer):
    def __init__(self, margin: float, validation_ratio: float,
                 weighting_method: SampleWeighting,
                 stopping_methods: list[StoppingMethod],
                 solver: Literal['dccp'] | None,
                 verbose: Literal[0, 1, 2],
                 random_state: np.random.RandomState) -> None:
        """
        Initialize the r-DEP trainer.

        param [any]:    see base class
        """

        super().__init__(margin, validation_ratio, weighting_method,
                         stopping_methods, solver, verbose, random_state)

        if self.solver != 'dccp':
            raise ValueError(f'Incompatibler solver for RDEP: {self.solver}')

    def at_training_start(self, data_dim: int) -> None:
        # similar to l-DEP, see corresponding files for implementation comments
        self._objective = None

        # Extracted parameters that will be populated during training but need
        # initial values for linearization
        self.max_perceptron = self.random_state.randn(data_dim)
        self.min_perceptron = self.random_state.randn(data_dim)
        self.lambda_ = self.random_state.randn()

        self._max_training_weights = cp.Variable(data_dim)
        self._min_training_weights = cp.Variable(data_dim)
        self._training_lambda = cp.Variable()

    def get_problem(self, X: np.ndarray, y: np.ndarray,
                    cost_weights: np.ndarray) -> cp.Problem:
        K = X.shape[0]

        # the objective and the slack variables do not change, cache them
        if self._objective is None:
            self._slack = cp.Variable(K)
            self._objective = cp.Minimize(
                    cp.sum(cp.multiply(cp.pos(self._slack), cost_weights)))

        idx_max = np.argmax(self.max_perceptron + X @ self.max_matrix.T,
                            axis=1)
        idx_min = np.argmin(self.min_perceptron + X @ self.min_matrix.T,
                            axis=1)

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

            expr_max = X_ @ self._max_training_matrix.T
            expr_min = X_ @ self._min_training_matrix.T

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

    def get_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        expr_max = np.max(self.max_perceptron + X, axis=1)
        expr_min = np.min(self.min_perceptron + X, axis=1)

        # can use the expressions directly without needing to multiply by lambda
        # since still in the training phase
        cost = self.margin + (expr_max + expr_min) * (1 - 2 * y)
        return np.maximum(0, cost).sum()

    def after_epoch(self) -> None:
        # update the perceptrons weights from this epoch's results
        if self._max_training_weights.value is None or \
                self._min_training_weights.value is None:
            raise ValueError('CvxPy could not optimize a perceptron')
        if self._training_lambda.value is None:
            raise ValueError('CvxPy could not optimize lambda')

        self.max_perceptron = self._max_training_weights.value
        self.min_perceptron = self._min_training_weights.value
        self.lambda_ = self._training_lambda.value

    def save_best(self) -> None:
        self.saved = {
            'max_w': np.copy(self.max_perceptron),
            'min_w': np.copy(self.min_perceptron),
            'lambda': self.lambda_,
        }

    def rollback_to_best(self) -> None:
        self.max_perceptron = self.saved['max_w']
        self.min_perceptron = self.saved['min_w']
        self.lambda_ = self.saved['lambda']
