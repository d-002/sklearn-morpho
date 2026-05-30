import numpy as np
import cvxpy as cp
from typing import Literal, cast
from warnings import warn

from .dccp_wrapper import DccpTrainer
from ..weighting import SampleWeighting
from ..stopping import StoppingMethod

class RDEPDccpTrainer(DccpTrainer):
    def __init__(self, lambda_bounds: tuple[float, float], margin: float,
                 validation_ratio: float, weighting_method: SampleWeighting,
                 stopping_methods: list[StoppingMethod],
                 solver: Literal['dccp'] | None,
                 verbose: Literal[0, 1, 2],
                 random_state: np.random.RandomState) -> None:
        """
        Initialize the r-DEP trainer.

        param lambda_bounds: A pair of min and max values for lambda, to avoid
                             solvers (especially dccp) from failing to optimize.
                             To keep the constraints at the right convexity,
                             the bounds must be inside [0, 1].
        param [others]:      See base class.
        """

        super().__init__(margin, validation_ratio, weighting_method,
                         stopping_methods, solver, verbose, random_state)

        if lambda_bounds[0] < 0 or lambda_bounds[1] > 1:
            raise ValueError('Invalid lambda_bounds, expected within [0, 1] '
                             f'got {list(lambda_bounds)}')
        if (lambda_bounds[0] == 0 or lambda_bounds[1] == 1) \
                and solver == 'dccp':
            warn('Warning: lambda_bounds may be inappropriate for dccp solver: '
                 f'{list(lambda_bounds)}')

        self.lambda_bounds = lambda_bounds

    def at_training_start(self, data_dim: int) -> None:
        # similar to l-DEP, see corresponding files for implementation comments
        self._objective = None

        # Extracted parameters that will be populated during training but need
        # initial values for linearization
        self.max_perceptron = self.random_state.randn(data_dim)
        self.min_perceptron = self.random_state.randn(data_dim)
        self.lambda_ = .5

        # to make the problem DCP, the training weights absorb the
        # multiplications involving lambda_, they are explicited at the end of
        # epochs
        self._max_training_weights = cp.Variable(data_dim)
        self._min_training_weights = cp.Variable(data_dim)
        self._training_lambda = cp.Variable()

    def set_objective(self, X: np.ndarray, y: np.ndarray,
                      cost_weights: np.ndarray) -> None:
        # the objective and the slack variables do not change, cache them
        if self._objective is not None:
            return

        # figure out whether we should invert the perceptron's output, since
        # the lower class should be lower in coordinates
        labels, inv, counts = np.unique(y, return_inverse=True,
                                        return_counts=True)
        sums = np.zeros((len(labels), X.shape[1]))
        np.add.at(sums, inv, X)
        centroids = sums / counts[:, np.newaxis]

        # set the objective
        self.invert_res = centroids[0].sum() > centroids[1].sum()

        self._slack = cp.Variable(len(X))

        self._objective = cp.Minimize(
            cp.sum(cp.multiply(cp.pos(self._slack), cost_weights))
        )

    def get_problem_unproven(self, X: np.ndarray, y: np.ndarray,
                             cost_weights: np.ndarray) -> cp.Problem:
        K = X.shape[0]
        self.set_objective(X, y, cost_weights)

        # Constraints: convex constraints are for data points in the first
        # class, while concave ones are for points in the second class.
        # Therefore, linearize things when needed to make the problem convex,
        # sometimes using values from the previous epoch.

        idx_max = np.argmax(self.lambda_ * X + self.max_perceptron, axis=1)
        idx_min = np.argmin((1 - self.lambda_) * X + self.min_perceptron,
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

            # add weights to every row of (X @ matrix.T) using np.ones to create
            # a matrix safely for cvxpy's cpp backend
            ones = np.ones((K_, 1))
            expr_max = self._training_lambda * X_
            expr_min = (1 - self._training_lambda) * X_
            expr_max += ones @ cp.reshape(self._max_training_weights,
                                          (1, self.max_perceptron.size),
                                          order='C')
            expr_min += ones @ cp.reshape(self._min_training_weights,
                                          (1, self.min_perceptron.size),
                                          order='C')

            active_max = cp.sum(cp.multiply(M_max[mask], expr_max), axis=1)
            active_min = cp.sum(cp.multiply(M_min[mask], expr_min), axis=1)

            if (label == 0) ^ self.invert_res:
                constraints.append(self.margin + cp.max(expr_max, axis=1)
                                   <= self._slack[mask] - active_min)
            else:
                constraints.append(self.margin - cp.min(expr_min, axis=1)
                                   <= self._slack[mask] + active_max)

        # avoid lambda making some constraints convex when they should be
        # concave and vice versa
        constraints += [self.lambda_bounds[0] <= self._training_lambda,
                        self._training_lambda <= self.lambda_bounds[1]]
        return cp.Problem(cast(cp.Minimize, self._objective), constraints)

    def get_problem_dccp(self, X: np.ndarray, y: np.ndarray,
                         cost_weights: np.ndarray) -> cp.Problem:
        self.set_objective(X, y, cost_weights)

        constraints = []
        for label in [0, 1]:
            mask = y == label
            if not np.any(mask):
                continue

            X_ = X[mask]
            K_ = X_.shape[0]

            ones = np.ones((K_, 1))
            expr_max = self._training_lambda * X_
            expr_min = (1 - self._training_lambda) * X_
            expr_max += ones @ cp.reshape(self._max_training_weights,
                                          (1, self.max_perceptron.size),
                                          order='C')
            expr_min += ones @ cp.reshape(self._min_training_weights,
                                          (1, self.min_perceptron.size),
                                          order='C')

            expr_max = cp.max(expr_max, axis=1)
            expr_min = cp.min(expr_min, axis=1)

            if (label == 0) ^ self.invert_res:
                constraints.append(self.margin + expr_max <=
                                   self._slack[mask] - expr_min)
            else:
                constraints.append(self.margin - expr_min <=
                                   self._slack[mask] + expr_max)

        constraints += [self.lambda_bounds[0] <= self._training_lambda,
                        self._training_lambda <= self.lambda_bounds[1]]
        return cp.Problem(cast(cp.Minimize, self._objective), constraints)

    def get_problem(self, X: np.ndarray, y: np.ndarray,
                    cost_weights: np.ndarray) -> cp.Problem:
        if self.solver == 'dccp':
            return self.get_problem_dccp(X, y, cost_weights)
        return self.get_problem_unproven(X, y, cost_weights)

    def get_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        expr_max = np.max(self.lambda_ * X + self.max_perceptron, axis=1)
        expr_min = np.min((1 - self.lambda_) * X + self.min_perceptron, axis=1)

        cost = (expr_max + expr_min) * (1 - 2 * y) * (1 - 2 * self.invert_res)

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

        self.max_perceptron /= self.lambda_
        self.min_perceptron /= self.lambda_

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
