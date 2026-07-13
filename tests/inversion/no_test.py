import numpy as np

from sklearn_morpho import inversion


def test_init() -> None:
    inversion.NoInversion()

def test_logic() -> None:
    heuristic = inversion.NoInversion()
    assert not heuristic.should_inverse(np.zeros(3), np.ones(5))
