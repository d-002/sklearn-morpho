import numpy as np
import cvxpy as cp
from typing import Literal, cast
from warnings import warn

from .dccp_wrapper import DccpTrainer
from ..weighting import SampleWeighting
from ..stopping import StoppingMethod

class SimplePerceptronDccpTrainer(DccpTrainer):
    """
    Simple perceptron trainer.

    During training, the final parameters are implicitly embedded into one
    another.
    Some of them (like lambda) are only extracted once the training ends.
    This is not necessary for performance, but is there for readability and
    interpretability reasons once the estimator is trained.
    """

    def __init__(self, kind: Literal['max', 'min'], margin: float,
                 penalty: float, validation_ratio: float,
                 weighting_method: SampleWeighting,
                 stopping_methods: list[StoppingMethod],
                 use_dccp_library: bool, verbose: Literal[0, 1, 2],
                 random_state: np.random.RandomState) -> None:
        """
        Initialize the r-DEP trainer.

        param kind:     Whether the perceptron is dilation or erosion.
        param [others]: See base class.
        """

        super().__init__(margin, penalty, validation_ratio, weighting_method,
                         stopping_methods, use_dccp_library, verbose,
                         random_state)

        self.kind = kind

    def at_training_start(self, data_dim: int) -> None:
        # similar to l-DEP, see corresponding files for implementation comments
        self._objective = None

        self.weights = self.random_state.randn(data_dim)
        self._training_weights = cp.Variable(data_dim)

    def set_objective(self, X: np.ndarray,
                      cost_weights: np.ndarray) -> None:
        # the objective and the slack variables do not change, cache them
        if self._objective is not None:
            return

        self._slack = cp.Variable(len(X))

        value = cp.sum(cp.multiply(cp.pos(self._slack), cost_weights))
        if self.penalty > 0:
            value += self.penalty * cp.sum_squares(self._training_weights)

        self._objective = cp.Minimize(value)

    def get_problem_linearized(self, X: np.ndarray, y: np.ndarray,
                               cost_weights: np.ndarray) -> cp.Problem:
        K = X.shape[0]
        self.set_objective(X, cost_weights)

        if self.kind == 'max':
            idx = np.argmax(X + self.weights, axis=1)
        else:
            idx = np.argmin(X + self.weights, axis=1)

        M = np.zeros((K, self.weights.size))
        M[np.arange(K), idx] = 1

        constraints = []
        for label in [0, 1]:
            mask = y == label
            if not np.any(mask):
                continue

            X_ = X[mask]
            K_ = X_.shape[0]

            # add weights to every row of (X @ matrix.T) using np.ones to create
            # a matrix safely for cvxpy's cpp backend
            ones = np.ones((K_, 1))
            expr = X_ + ones @ cp.reshape(self._training_weights,
                                          (1, self.weights.size),
                                          order='C')

            active_elt = cp.sum(cp.multiply(M[mask], expr), axis=1)

            if self.kind == 'max':
                if label == 0:
                    constraints.append(self.margin + cp.max(expr, axis=1)
                                       <= self._slack[mask])
                else:
                    constraints.append(self.margin
                                       <= self._slack[mask] + active_elt)
            else:
                if label == 0:
                    constraints.append(self.margin + active_elt
                                       <= self._slack[mask])
                else:
                    constraints.append(self.margin - cp.min(expr, axis=1)
                                       <= self._slack[mask])


        return cp.Problem(cast(cp.Minimize, self._objective), constraints)

    def get_problem_dccp(self, X: np.ndarray, y: np.ndarray,
                         cost_weights: np.ndarray) -> cp.Problem:
        self.set_objective(X, cost_weights)

        constraints = []
        for label in [0, 1]:
            mask = y == label
            if not np.any(mask):
                continue

            X_ = X[mask]
            K_ = X_.shape[0]

            ones = np.ones((K_, 1))
            expr = X_ + ones @ cp.reshape(self._training_weights,
                                          (1, self.weights.size),
                                          order='C')

            if self.kind == 'max':
                expr = cp.max(expr, axis=1)
            else:
                expr = cp.min(expr, axis=1)

            if label == 0:
                constraints.append(self.margin + expr <= self._slack[mask])
            else:
                constraints.append(self.margin - expr <= self._slack[mask])

        return cp.Problem(cast(cp.Minimize, self._objective), constraints)

    def get_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.kind == 'max':
            expr = np.max(X + self.weights, axis=1)
        else:
            expr = np.min(X + self.weights, axis=1)

        cost = (expr) * (1 - 2 * y)

        return np.maximum(0, cost).sum()

    def after_epoch(self) -> None:
        # update the perceptrons weights from this epoch's results
        if self._training_weights.value is None:
            raise ValueError('CvxPy could not optimize the perceptron')

        self.weights = self._training_weights.value

    def save_best(self) -> None:
        self.saved = {
            'weights': np.copy(self.weights),
        }

    def rollback_to_best(self) -> None:
        self.weights = self.saved['weights']
