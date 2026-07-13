import numpy as np
from abc import ABC, abstractmethod

class InversionHeuristic(ABC):
    """
    Base class for inversion heuristics and binary datasets.
    Used by classifiers that can only handle datasets where one class is in a
    specific position compared to the other.
    """

    @abstractmethod
    def should_inverse(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Returns whether, according to this heuristic, the two target classes
        should be inverted to benefit the classifier.
        """
