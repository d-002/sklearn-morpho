from typing import Any
import numpy as np
import cvxpy as cp

from .dccp_wrapper import DccpTrainer
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

        super().__init__([self.max_perceptron, self.min_perceptron], weighted,
                         max_iterations, done_threshold, verbose)

    def at_training_start(self) -> list[cp.Constraint]:
        super().at_training_start()
        self._training_lambda = cp.Variable()

        # will be populated during training, but need an initial value for
        # linearization
        self.lambda_ = np.random.rand()

        return [self._training_lambda >= 0, self._training_lambda <= 1]

    def cvx_cost_function(self, weights: list[cp.Variable], x: np.ndarray,
                          y: Any, slack: cp.Variable,
                          k: int) -> cp.Constraint | None:
        if y != 0:
            return None

        # manual linear approximation for speed, using previously calculated
        # perceptron weights and lambda, but it should still converge
        index = np.argmin((1 - self.lambda_) *
                          (self.min_perceptron.weights + x))

        # use absorbed values max_weights = max_weights * lambda for cvxpy
        # compliance, inspired by arXiv:2011.06512v1
        max_weights, min_weights = weights
        value = cp.max(max_weights + self._training_lambda * x) + \
                (min_weights + (1 - self._training_lambda) * x)[index]
        return slack[k] >= value

    def ccv_cost_function_made_convex(self, weights: list[cp.Variable],
                                      x: np.ndarray,
                                      y: Any, slack: cp.Variable,
                                      k: int) -> cp.Constraint | None:
        if y != 1:
            return None

        index = np.argmax(self.lambda_ * (self.max_perceptron.weights + x))

        max_weights, min_weights = weights
        value = -(max_weights + self._training_lambda * x)[index] - \
                cp.min(min_weights + (1 - self._training_lambda) * x)
        return slack[k] >= value

    def after_training_iteration(self, optimized_weights: list[cp.Variable]
                                 ) -> None:
        # update the perceptrons weights
        for perceptron, weights in zip(self.perceptrons, optimized_weights):
            if weights.value is None:
                raise ValueError('CvxPy could not optimize a perceptron')
            perceptron.weights = weights.value

        # also update lambda_ from its current value
        if self._training_lambda.value is None:
            raise ValueError('CvxPy could not optimize lambda')
        self.lambda_ = self._training_lambda.value

    def at_training_end(self) -> None:
        # recover the actual morphological weights from the absorbed values
        print(self.lambda_)
        self.max_perceptron.weights /= self.lambda_
        self.min_perceptron.weights /= (1 - self.lambda_)
