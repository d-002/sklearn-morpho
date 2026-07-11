import numpy as np
import pytest

from sklearn_morpho import weighting


def test_init():
    weighting.DistSampleWeighting()


def test_fit_single_elements():
    X = np.array([(-1, 0), (1, 0)])
    y = np.array([-1, 1])

    w = weighting.DistSampleWeighting()
    w.fit(X, y)


def test_fit_transform_single_elements():
    X = np.array([(-1, 0), (1, 0)])
    y = np.array([-1, 1])

    w = weighting.DistSampleWeighting()
    w.fit(X, y)

    w = weighting.DistSampleWeighting()
    weights, cost_normalizer = w.fit_transform(X, y)

    assert np.allclose(weights, np.ones(2))
    assert np.isclose(cost_normalizer, 1)


def test_transform_not_allowed():
    w = weighting.DistSampleWeighting()
    with pytest.raises(ValueError):
        w.transform(np.zeros(10))
