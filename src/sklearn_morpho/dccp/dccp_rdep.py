from typing import Literal
import numpy as np
import cvxpy as cp

from .dccp_wrapper import DccpTrainer
from ..weighting import SampleWeighting
from ..stopping import StoppingMethod

class RDEPDccpTrainer(DccpTrainer):
    def __init__(self, _lambda: float, margin: float, validation_ratio: float,
                 weighting_method: SampleWeighting,
                 stopping_methods: list[StoppingMethod],
                 solver: Literal['dccp'] | None,
                 verbose: Literal[0, 1, 2],
                 random_state: np.random.RandomState) -> None:
        """
        Initialize the r-DEP trainer.

        param _lambda:     Fixed lambda parameter.
                           Not learnable at this point, must use techniques like
                           cross-validation to tune.
        param [others]:    See base class.
        """

        super().__init__(margin, validation_ratio, weighting_method,
                         stopping_methods, solver, verbose, random_state)

        self._lambda = _lambda

        if not 0 <= _lambda <= 1:
            raise ValueError('Invalid lambda, expected >= 0 and <= 1 '
                             f'but got {_lambda}')
        # TODO remove comments?
        #if self.solver != 'dccp':
        #    raise ValueError(f'Incompatibler solver for RDEP: {self.solver}')

    def at_training_start(self, data_dim: int) -> None:
        # similar to l-DEP, see corresponding files for implementation comments
        self._objective = None

        # Extracted parameters that will be populated during training but need
        # initial values for linearization
        self.max_perceptron = self.random_state.randn(data_dim)
        self.min_perceptron = self.random_state.randn(data_dim)

        self._max_training_weights = cp.Variable(data_dim)
        self._min_training_weights = cp.Variable(data_dim)

    def get_problem_unproven(self, X: np.ndarray, y: np.ndarray,
                             cost_weights: np.ndarray) -> cp.Problem:
        K = X.shape[0]

        # the objective and the slack variables do not change, cache them
        if self._objective is None:
            self._slack = cp.Variable(K)
            penalty = 1e-6 * (cp.sum_squares(self._max_training_weights) +
                              cp.sum_squares(self._min_training_weights))

            self._objective = cp.Minimize(
                cp.sum(cp.multiply(cp.pos(self._slack), cost_weights)) +
                0#penalty
            )

        # Constraints: convex constraints are for data points in the first
        # class, while concave ones are for points in the second class.
        # Therefore, linearize things when needed to make the problem convex,
        # sometimes using values from the previous epoch.

        # WARNING (TODO remove by implementing): if making lambda learnable, make sure to add it here as well because it could be negative

        idx_max = np.argmax(self.max_perceptron + X, axis=1)
        idx_min = np.argmin(self.min_perceptron + X, axis=1)

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

            # add weights to every row of (X @ matrix.T) using np.ones to create
            # a matrix safely for cvxpy's cpp backend
            ones = np.ones((K_, 1))
            expr_max = X_ + ones @ cp.reshape(self._max_training_weights,
                                              (1, self.max_perceptron.size),
                                              order='C')
            expr_min = X_ + ones @ cp.reshape(self._min_training_weights,
                                              (1, self.min_perceptron.size),
                                              order='C')

            active_max = cp.sum(cp.multiply(M_max[mask], expr_max), axis=1)
            active_min = cp.sum(cp.multiply(M_min[mask], expr_min), axis=1)

            if label == 0:
                constraints.append(self.margin + cp.max(expr_max, axis=1)
                                   <= self._slack[mask] - active_min)
            else:
                constraints.append(self.margin - cp.min(expr_min, axis=1)
                                   <= self._slack[mask] + active_max)
        return cp.Problem(self._objective, constraints)

    def get_problem_dccp(self, X: np.ndarray, y: np.ndarray,
                         cost_weights: np.ndarray) -> cp.Problem:
        K = X.shape[0]

        # the objective and the slack variables do not change, cache them
        if self._objective is None:
            self._slack = cp.Variable(K)
            penalty = 1e-6 * (cp.sum_squares(self._max_training_weights) +
                              cp.sum_squares(self._min_training_weights))

            self._objective = cp.Minimize(
                cp.sum(cp.multiply(cp.pos(self._slack), cost_weights)) +
                penalty
            )

        constraints = []
        for label in [0, 1]:
            mask = y == label
            if not np.any(mask):
                continue

            X_ = X[mask]
            K_ = X_.shape[0]

            ones = np.ones((K_, 1))
            expr_max = X_ + ones @ cp.reshape(self._max_training_weights,
                                              (1, self.max_perceptron.size),
                                              order='C')
            expr_min = X_ + ones @ cp.reshape(self._min_training_weights,
                                              (1, self.min_perceptron.size),
                                              order='C')
            expr_max = self._lambda * cp.max(expr_max, axis=1)
            expr_min = (1 - self._lambda) * cp.min(expr_min, axis=1)

            if label == 0:
                constraints.append(self.margin + expr_max <=
                                   self._slack[mask] - expr_min)
            else:
                constraints.append(self.margin - expr_min <=
                                   self._slack[mask] + expr_max)
        return cp.Problem(self._objective, constraints)

    def get_problem(self, X: np.ndarray, y: np.ndarray,
                    cost_weights: np.ndarray) -> cp.Problem:
        if self.solver == 'dccp':
            return self.get_problem_dccp(X, y, cost_weights)
        return self.get_problem_unproven(X, y, cost_weights)

    def get_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        expr_max = self._lambda * np.max(self.max_perceptron + X, axis=1)
        expr_min = (1 - self._lambda) * np.min(self.min_perceptron + X, axis=1)

        cost = (expr_max + expr_min) * (1 - 2 * y)

        return np.maximum(0, cost).sum()

    def after_epoch(self, X, y) -> None: # TODO remove arguments
        # update the perceptrons weights from this epoch's results
        if self._max_training_weights.value is None or \
                self._min_training_weights.value is None:
            raise ValueError('CvxPy could not optimize a perceptron')

        self.max_perceptron = self._max_training_weights.value
        self.min_perceptron = self._min_training_weights.value

        # TODO remove
        print(self.max_perceptron)
        print(self.min_perceptron)
        expr_max = self._lambda * np.max(self.max_perceptron + X, axis=1)
        expr_min = (1 - self._lambda) * np.min(self.min_perceptron + X, axis=1)
        res = expr_max + expr_min > 0
        print(f'Training accuracy: {np.count_nonzero(res ^ y)/len(y)*100.2}%')

    def save_best(self) -> None:
        self.saved = {
            'max_w': np.copy(self.max_perceptron),
            'min_w': np.copy(self.min_perceptron),
        }

    def rollback_to_best(self) -> None:
        self.max_perceptron = self.saved['max_w']
        self.min_perceptron = self.saved['min_w']
