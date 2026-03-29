import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

class Perceptron(ABC):
    """
    Morphological perceptron for data classification.
    Abstract class, see MaxPerceptron and MinPerceptron.
    """

    def __init__(self, weights: np.ndarray | int,
                 dtype: npt.DTypeLike = np.float64) -> None:
        """
        Initialize a tropical perceptron.
        Weights are of the provided dtype.
        If param weights is an integer, use it as the size of zero weights.
        """

        if type(weights) == int:
            self.weights = np.zeros(weights, dtype=dtype)
        else:
            self.weights = np.array(weights, dtype=dtype)

        self.dim = self.weights.shape[0]
        self.dtype = np.dtype(dtype)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(w[{self.dim}] = {self.weights})'

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.floating :
        """
        Forward pass for the perceptron.
        """

class MaxPerceptron(Perceptron):
    def forward(self, inputs: np.ndarray) -> np.floating :
        return np.max(self.weights + inputs)

class MinPerceptron(Perceptron):
    def forward(self, inputs: np.ndarray) -> np.floating :
        return np.min(self.weights + inputs)
