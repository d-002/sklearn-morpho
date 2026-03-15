from typing import Any
import numpy as np
import cvxpy as cp

from .dccp import DccpTrainer
from ..perceptron import MaxPerceptron, MinPerceptron

class DepDccpTrainer(DccpTrainer):
    def __init__(self, N: int, _lambda: float, weighted: bool,
                 max_iterations: int = 100, done_threshold = 1e-6,
                 verbose: bool = False) -> None:
        """
        Initialize the dilation-erosion perceptron trainer.

        param N:        The dimension of the data.
        param [others]: See base class.
        """

        if _lambda < 0 or _lambda > 1:
            raise ValueError(f'lambda must be between 0 and 1, got {_lambda}')

        self.max_perceptron = MaxPerceptron(N)
        self.min_perceptron = MinPerceptron(N)
        self._lambda = _lambda

        super().__init__([self.max_perceptron, self.min_perceptron], weighted,
                         max_iterations, done_threshold, verbose)

    def at_training_start(self) -> None:
        """
        Initialize all perceptrons before training.
        """

        for perceptron in self.perceptrons:
            perceptron.weights = np.zeros(perceptron.dim)
            perceptron.bias = perceptron.get_neutral_bias()

    def _forward(self, x1: cp.Expression, x2: cp.Expression) -> cp.Expression:
        return self._lambda * x1 + (1 - self._lambda) * x2

    def cvx_cost_function(self, weights: list[cp.Variable], x: cp.Variable,
                          y: Any, slack: cp.Variable,
                          k: int) -> cp.Constraint | None:
        if y != 0:
            return None

        # use weights from the perceptron, this makes dccp converge slower but
        # argmax is not available directly in cvxpy
        index = np.argmin(self.min_perceptron.weights + x)

        max_weights, min_weights = weights
        value = self._forward(cp.max(max_weights + x), min_weights[index] + x)
        return slack[k] >= value

    def ccv_cost_function_made_convex(self, weights: list[cp.Variable],
                                      x: cp.Variable,
                                      y: Any, slack: cp.Variable,
                                      k: int) -> cp.Constraint | None:
        if y != 1:
            return None

        index = np.argmax(self.max_perceptron.weights + x)

        max_weights, min_weights = weights
        value = -self._forward(max_weights[index] + x, cp.min(min_weights + x))
        return slack[k] >= value
