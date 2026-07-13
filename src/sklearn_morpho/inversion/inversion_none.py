import numpy as np

from .inversion_base import InversionHeuristic

class NoInversion(InversionHeuristic):
    """
    This heuristic does no inversion no matter what.
    """

    def should_inverse(self, X: np.ndarray, y: np.ndarray) -> bool:
        return False
