import pytest
import numpy as np

from sklearn_morpho import inversion


def test_init() -> None:
    inversion.CentroidInversion(np.ones(1))

def test_data_size() -> None:
    inversion.CentroidInversion(np.ones(1))

    heuristic = inversion.CentroidInversion(np.ones(2))
    y = np.repeat([0, 1], 5)
    heuristic.should_inverse(np.ones((10, 2)), y)
    with pytest.raises(ValueError):
        heuristic.should_inverse(np.ones((10, 3)), y)

def test_logic() -> None:
    heuristic = inversion.CentroidInversion(np.ones(2))
    X = np.repeat([[-1, -1], [1, 1]], 5, axis=0)
    y = np.repeat([0, 1], 5)
    assert not heuristic.should_inverse(X, y)
    assert heuristic.should_inverse(-X, y)
