import pytest
import numpy as np

from friendly_dataset import friendly_dataset
from sklearn.metrics import f1_score

from sklearn_morpho.weighting import DistSampleWeighting
from sklearn_morpho.stopping import StoppingMethod, EpochStoppingMethod
from sklearn_morpho import LDEP


def test_init():
    LDEP()


def test_train_noverif():
    ldep = LDEP()

    X, y = friendly_dataset()
    ldep.fit(X, y)


def test_train_zero_matrices():
    ldep = LDEP()

    X, y = np.zeros((2, 1)), np.arange(2)
    with pytest.raises(ValueError):
        ldep.fit(X, y)


def test_train():
    ldep = LDEP()

    X, y = friendly_dataset()
    ldep.fit(X, y)

    assert f1_score(y, ldep.predict(X)) >= 0.8


def test_train_custom_weighted_stopping():
    stopping_methods: list[StoppingMethod] = [EpochStoppingMethod()]
    ldep = LDEP(
        weighting_method=DistSampleWeighting(),
        stopping_methods=stopping_methods,
    )

    X, y = friendly_dataset()
    ldep.fit(X, y)

    assert f1_score(y, ldep.predict(X)) >= 0.8


def test_train_with_penalty():
    ldep = LDEP(penalty=0.1)

    X, y = friendly_dataset()
    ldep.fit(X, y)

    assert f1_score(y, ldep.predict(X)) >= 0.8


def test_train_dccp():
    ldep = LDEP(use_dccp_library=True)

    X, y = friendly_dataset()
    ldep.fit(X, y)

    assert f1_score(y, ldep.predict(X)) >= 0.8


def test_train_dccp_with_penalty():
    ldep = LDEP(penalty=0.1, use_dccp_library=True)

    X, y = friendly_dataset()
    ldep.fit(X, y)

    assert f1_score(y, ldep.predict(X)) >= 0.8
