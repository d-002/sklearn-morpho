import cvxpy as cp
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

class Perceptron(ABC):
    """
    Perceptron with bias for data classification.
    Abstract class, see MaxPerceptron and MinPerceptron.
    """

    def __init__(self, weights: np.ndarray | int,
                 bias: np.floating | None = None,
                 dtype: npt.DTypeLike = np.float64) -> None:
        """
        Initialize a tropical perceptron.
        If param weights is an integer, use it to initialize the weights
        (all 0s, of dimension `weights`).
        Weights are of the provided dtype.
        If no bias is provided, use the overriden get_neutral_bias method.
        """

        if bias is None:
            self.bias = self.get_neutral_bias()
        else:
            self.bias = bias

        if type(weights) == int:
            self.weights = np.zeros(weights, dtype=dtype)
        else:
            self.weights = np.array(weights, dtype=dtype)

        self.dim = self.weights.size
        self.dtype = np.dtype(dtype)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(b = {self.bias}, ' \
                f'w[{self.dim}] = {self.weights})'

    @abstractmethod
    def get_neutral_bias(self) -> np.floating:
        pass

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.floating :
        pass

class MaxPerceptron(Perceptron):
    def get_neutral_bias(self) -> np.floating:
        return np.float64(-np.inf)

    def forward(self, inputs: np.ndarray) -> np.floating :
        return max(self.bias, np.max(self.weights + inputs))

class MinPerceptron(Perceptron):
    def get_neutral_bias(self) -> np.floating:
        return np.float64(np.inf)

    def forward(self, inputs: np.ndarray) -> np.floating :
        return min(self.bias, np.min(self.weights + inputs))
