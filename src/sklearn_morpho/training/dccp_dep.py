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
        self.lambda_ = 0 # TODO, TEMPORARY

        super().__init__([self.max_perceptron, self.min_perceptron], weighted,
                         max_iterations, done_threshold, verbose)

    def initialize_perceptrons(self) -> None:
        for perceptron in self.perceptrons:
            perceptron.weights = np.zeros(perceptron.dim)
            perceptron.bias = perceptron.get_neutral_bias()

    def cvx_cost_function(self, weights: list[cp.Variable], x: cp.Variable,
                          y: Any, slack: cp.Variable,
                          k: int) -> cp.Constraint | None:
        pass

    def ccv_cost_function_made_convex(self, weights: list[cp.Variable],
                                      x: cp.Variable,
                                      y: Any, slack: cp.Variable,
                                      k: int) -> cp.Constraint | None:
        pass
