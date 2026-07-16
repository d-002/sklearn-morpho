from typing import Literal, cast
from warnings import warn

import cvxpy as cp
import numpy as np

from ..inversion import InversionHeuristic
from ..stopping import StoppingMethod
from ..weighting import SampleWeighting
from .dccp_wrapper import DccpTrainer


class SavedState:
    max_w: np.ndarray
    min_w: np.ndarray
    lambda_: float

    def __init__(
        self, max_w: np.ndarray, min_w: np.ndarray, lambda_: float
    ) -> None:
        self.max_w = max_w
        self.min_w = min_w
        self.lambda_ = lambda_


class DEPDccpTrainer(DccpTrainer):
    """
DEP trainer.

During training, the final parameters are implicitly embedded into one another.
Some of them (like lambda) are only extracted once the training ends.
This is not necessary for performance, but is there for readability and
interpretability reasons once the estimator is trained.
    """

    def __init__(
        self,
        lambda_bounds: tuple[float, float],
        margin: float,
        penalty: float,
        validation_ratio: float,
        weighting_method: SampleWeighting,
        stopping_methods: list[StoppingMethod],
        inversion_method: InversionHeuristic,
        solver: str | None,
        verbose: Literal[0, 1, 2],
        random_state: np.random.RandomState,
    ) -> None:
        """
Initialize the DEP trainer.

- param `lambda_bounds`:
  A pair of min and max values for lambda, to avoid solvers (especially dccp)
  from failing to optimize.
  To keep the constraints at the right convexity, the bounds must be inside
  [0, 1].
- param `inversion_method`:
  The heuristic to use to know whether to invert the target classes, as the
  dataset's orientation might not always be favorable.
- param `[others]`:
  See base class.
        """

        super().__init__(
            margin,
            penalty,
            validation_ratio,
            weighting_method,
            stopping_methods,
            solver,
            verbose,
            random_state,
        )

        if lambda_bounds[0] < 0 or lambda_bounds[1] > 1:
            raise ValueError(
                'Invalid lambda_bounds, expected within [0, 1] '
                f'got {list(lambda_bounds)}'
            )
        if (lambda_bounds[0] == 0 or lambda_bounds[1] == 1) and solver:
            warn(
                'Warning: lambda_bounds may be inappropriate for dccp solver: '
                f'{list(lambda_bounds)}'
            )

        self.lambda_bounds = lambda_bounds
        self.inversion_method = inversion_method

    def at_training_start(self, data_dim: int) -> None:
        # similar to l-DEP, see corresponding files for implementation comments
        self._objective: cp.Minimize | None = None

        # Extracted parameters that will be populated during training but need
        # initial values for linearization
        self.max_perceptron = self.random_state.randn(data_dim)
        self.min_perceptron = self.random_state.randn(data_dim)
        self.lambda_ = 0.5

        # to make the problem DCP, the training weights absorb the
        # multiplications involving lambda_, they are explicited at the end of
        # epochs
        self._max_training_weights = cp.Variable(data_dim)
        self._min_training_weights = cp.Variable(data_dim)
        self._training_lambda = cp.Variable()

    def set_objective(
        self, X: np.ndarray, y: np.ndarray, cost_weights: np.ndarray
    ) -> None:
        # the objective and the slack variables do not change, cache them
        if self._objective is not None:
            return

        n_features = len(np.unique(y))
        if n_features != 2:
            raise ValueError(
                'Detected degenerate dataset, perhaps after '
                'train/validation split. Expected 2 features, '
                f'found only {n_features} feature(s)'
            )

        self.invert_res = self.inversion_method.should_invert(X, y)

        self._slack = cp.Variable(len(X))

        value = cp.sum(cp.multiply(cp.pos(self._slack), cost_weights))
        if self.penalty > 0:
            value += self.penalty * (
                cp.sum_squares(self._max_training_weights)
                + cp.sum_squares(self._min_training_weights)
            )

        self._objective = cp.Minimize(value)

    def get_problem_linearized(
        self, X: np.ndarray, y: np.ndarray, cost_weights: np.ndarray
    ) -> cp.Problem:
        K = X.shape[0]
        self.set_objective(X, y, cost_weights)

        # Constraints: convex constraints are for data points in the first
        # class, while concave ones are for points in the second class.
        # Therefore, linearize things when needed to make the problem convex,
        # sometimes using values from the previous epoch.

        idx_max = np.argmax(self.lambda_ * X + self.max_perceptron, axis=1)
        idx_min = np.argmin(
            (1 - self.lambda_) * X + self.min_perceptron, axis=1
        )

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
            expr_max += ones @ cp.reshape(
                self._max_training_weights,
                (1, self.max_perceptron.size),
                order='C',
            )
            expr_min += ones @ cp.reshape(
                self._min_training_weights,
                (1, self.min_perceptron.size),
                order='C',
            )

            active_max = cp.sum(cp.multiply(M_max[mask], expr_max), axis=1)
            active_min = cp.sum(cp.multiply(M_min[mask], expr_min), axis=1)

            if (label == 0) ^ self.invert_res:
                constraints.append(
                    self._slack[mask] - active_min
                    >= self.margin + cp.max(expr_max, axis=1)
                )
            else:
                constraints.append(
                    self._slack[mask] + active_max
                    >= self.margin - cp.min(expr_min, axis=1)
                )

        # avoid lambda making some constraints convex when they should be
        # concave and vice versa
        constraints += [
            self.lambda_bounds[0] <= self._training_lambda,
            self._training_lambda <= self.lambda_bounds[1],
        ]
        return cp.Problem(cast(cp.Minimize, self._objective), constraints)

    def get_problem_dccp(
        self, X: np.ndarray, y: np.ndarray, cost_weights: np.ndarray
    ) -> cp.Problem:
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
            expr_max += ones @ cp.reshape(
                self._max_training_weights,
                (1, self.max_perceptron.size),
                order='C',
            )
            expr_min += ones @ cp.reshape(
                self._min_training_weights,
                (1, self.min_perceptron.size),
                order='C',
            )

            expr_max = cp.max(expr_max, axis=1)
            expr_min = cp.min(expr_min, axis=1)

            if (label == 0) ^ self.invert_res:
                constraints.append(
                    self.margin + expr_max <= self._slack[mask] - expr_min
                )
            else:
                constraints.append(
                    self.margin - expr_min <= self._slack[mask] + expr_max
                )

        constraints += [
            self.lambda_bounds[0] <= self._training_lambda,
            self._training_lambda <= self.lambda_bounds[1],
        ]
        return cp.Problem(cast(cp.Minimize, self._objective), constraints)

    def get_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        expr_max = np.max(self.lambda_ * X + self.max_perceptron, axis=1)
        expr_min = np.min((1 - self.lambda_) * X + self.min_perceptron, axis=1)

        cost = (expr_max + expr_min) * (1 - 2 * y) * (1 - 2 * self.invert_res)

        res: float = np.maximum(0, cost).sum()
        return res

    def after_epoch(self) -> None:
        # update the perceptrons weights from this epoch's results
        if (
            self._max_training_weights.value is None
            or self._min_training_weights.value is None
        ):
            raise ValueError('CvxPy could not optimize a perceptron')
        if self._training_lambda.value is None:
            raise ValueError('CvxPy could not optimize lambda')

        self.max_perceptron = self._max_training_weights.value
        self.min_perceptron = self._min_training_weights.value
        self.lambda_ = cast(float, self._training_lambda.value)

        # if lambda is close to a number that creates divisions by zero, it is
        # safe to nullify the affected elements, that will not contribute anyway
        if np.isclose(self.lambda_, 0):
            self.max_perceptron = np.zeros_like(self.max_perceptron)
        else:
            self.max_perceptron /= self.lambda_

        if np.isclose(self.lambda_, 1):
            self.min_perceptron = np.zeros_like(self.min_perceptron)
        else:
            self.min_perceptron /= 1 - self.lambda_

    def save_best(self) -> None:
        self.saved = SavedState(
            np.copy(self.max_perceptron),
            np.copy(self.min_perceptron),
            self.lambda_,
        )

    def rollback_to_best(self) -> None:
        self.max_perceptron = self.saved.max_w
        self.min_perceptron = self.saved.min_w
        self.lambda_ = self.saved.lambda_
