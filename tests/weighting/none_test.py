import pytest
import numpy as np

from sklearn_morpho import weighting

def test_init():
    weighting.NoneSampleWeighting()

def test_fit_empty():
    X, y = np.arange(0, 2), np.arange(0)

    w = weighting.NoneSampleWeighting()
    w.fit(X, y)

def test_fit_nonempty():
    n = 100
    X, y = np.arange(n, 2), np.arange(n)

    w = weighting.NoneSampleWeighting()
    w.fit(X, y)

def test_fit_transform_empty():
    X, y = np.zeros(0), np.zeros(0)

    w = weighting.NoneSampleWeighting()
    weights, cost_normalizer = w.fit_transform(X, y)

    assert np.allclose(weights, np.zeros(0))
    assert np.isclose(cost_normalizer, 1)

def test_fit_transform_nonempty():
    n = 100
    X, y = np.zeros((n, 2)), np.zeros(n)

    w = weighting.NoneSampleWeighting()
    weights, cost_normalizer = w.fit_transform(X, y)

    assert np.allclose(weights, np.ones(n))
    assert np.isclose(cost_normalizer, 1)

def test_transform_not_allowed():
    w = weighting.NoneSampleWeighting()
    with pytest.raises(ValueError):
        w.transform(np.zeros(10))
