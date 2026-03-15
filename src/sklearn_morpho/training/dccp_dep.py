from typing import Any
import numpy as np
import cvxpy as cp

from .dccp import DccpTrainer
from ..perceptron import MaxPerceptron, MinPerceptron

class DepDccpTrainer(DccpTrainer):
    def __init__(self, N: int, weighted: bool,
                 max_iterations: int = 100, done_threshold = 1e-6,
                 verbose: bool = False) -> None:
        """
        Initialize the dilation-erosion perceptron trainer.

        param N:        The dimension of the data.
        param [others]: See base class.
        """

        self.max_perceptron = MaxPerceptron(N)
        self.min_perceptron = MinPerceptron(N)
        self.lambda_ = .5 # TODO, TEMPORARY, also division by 0 errors if lambda in {0, 1}
        # TODO fix lower / upper class to support all class types, then allow sklearn to classify again

        super().__init__([self.max_perceptron, self.min_perceptron], weighted,
                         max_iterations, done_threshold, verbose)

    def initialize_perceptrons(self) -> None:
        for perceptron in self.perceptrons:
            perceptron.weights = np.zeros(perceptron.dim)
            perceptron.bias = perceptron.get_neutral_bias()

    def _forward(self, x1: cp.Expression, x2: cp.Expression) -> cp.Expression:
        return self.lambda_ * x1 + (1 - self.lambda_) * x2

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

        # use weights from the perceptron, this makes dccp converge slower but
        # argmax is not available directly in cvxpy
        index = np.argmax(self.max_perceptron.weights + x)

        max_weights, min_weights = weights
        value = -self._forward(max_weights[index] + x, cp.min(min_weights + x))
        return slack[k] >= value
