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

        if not 0 <= _lambda < 1:
            raise ValueError('Invalid lambda, expected >= 0 and < 1 '
                             f'but got {_lambda}')
        if self.solver != 'dccp':
            raise ValueError(f'Incompatibler solver for RDEP: {self.solver}')

    def at_training_start(self, data_dim: int) -> None:
        # similar to l-DEP, see corresponding files for implementation comments
        self._objective = None

        # Extracted parameters that will be populated during training but need
        # initial values for linearization
        self.max_perceptron = self.random_state.randn(data_dim)
        self.min_perceptron = self.random_state.randn(data_dim)

        self._max_training_weights = cp.Variable(data_dim)
        self._min_training_weights = cp.Variable(data_dim)

    def get_problem(self, X: np.ndarray, y: np.ndarray,
                    cost_weights: np.ndarray) -> cp.Problem:
        K = X.shape[0]

        # the objective and the slack variables do not change, cache them
        if self._objective is None:
            self._slack = cp.Variable(K)
            self._objective = cp.Minimize(
                    cp.sum(cp.multiply(cp.pos(self._slack), cost_weights)))

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

            if label == 0:
                constraints.append(self.margin + cp.max(expr_max, axis=1) <=
                                   self._slack[mask] - cp.min(expr_min, axis=1))
            else:
                constraints.append(self.margin - cp.min(expr_min, axis=1) <=
                                   self._slack[mask] + cp.max(expr_max, axis=1))
        return cp.Problem(self._objective, constraints)

    def get_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        expr_max = self._lambda * np.max(self.max_perceptron + X, axis=1)
        expr_min = (1 - self._lambda) * np.min(self.min_perceptron + X, axis=1)

        cost = self.margin + (expr_max + expr_min) * (1 - 2 * y)
        return np.maximum(0, cost).sum()

    def after_epoch(self) -> None:
        # update the perceptrons weights from this epoch's results
        if self._max_training_weights.value is None or \
                self._min_training_weights.value is None:
            raise ValueError('CvxPy could not optimize a perceptron')

        self.max_perceptron = self._max_training_weights.value
        self.min_perceptron = self._min_training_weights.value

    def save_best(self) -> None:
        self.saved = {
            'max_w': np.copy(self.max_perceptron),
            'min_w': np.copy(self.min_perceptron),
        }

    def rollback_to_best(self) -> None:
        self.max_perceptron = self.saved['max_w']
        self.min_perceptron = self.saved['min_w']
