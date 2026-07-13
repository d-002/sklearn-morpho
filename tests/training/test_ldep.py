import numpy as np
import pytest
from friendly_dataset import friendly_dataset
from sklearn.metrics import f1_score

from sklearn_morpho import LDEP
from sklearn_morpho.stopping import EpochStoppingMethod, StoppingMethod
from sklearn_morpho.weighting import DistSampleWeighting


def test_init() -> None:
    LDEP()


def test_train_noverif() -> None:
    ldep = LDEP()

    X, y = friendly_dataset()
    ldep.fit(X, y)


def test_train_zero_matrices() -> None:
    ldep = LDEP()

    X, y = np.zeros((2, 1)), np.arange(2)
    with pytest.raises(ValueError):
        ldep.fit(X, y)


def test_train() -> None:
    X, y = friendly_dataset()

    for use_dccp_library in [False, True]:
        ldep = LDEP(use_dccp_library=use_dccp_library)
        ldep.fit(X, y)

        assert f1_score(y, ldep.predict(X)) >= 0.8


def test_train_custom_weighted_stopping() -> None:
    stopping_methods: list[StoppingMethod] = [EpochStoppingMethod()]
    ldep = LDEP(
        weighting_method=DistSampleWeighting(),
        stopping_methods=stopping_methods,
    )

    X, y = friendly_dataset()
    ldep.fit(X, y)

    assert f1_score(y, ldep.predict(X)) >= 0.8


def test_train_with_penalty() -> None:
    ldep = LDEP(penalty=0.1)

    X, y = friendly_dataset()
    ldep.fit(X, y)

    assert f1_score(y, ldep.predict(X)) >= 0.8


def test_train_dccp() -> None:
    ldep = LDEP(use_dccp_library=True)

    X, y = friendly_dataset()
    ldep.fit(X, y)

    assert f1_score(y, ldep.predict(X)) >= 0.8


def test_train_dccp_with_penalty() -> None:
    ldep = LDEP(penalty=0.1, use_dccp_library=True)

    X, y = friendly_dataset()
    ldep.fit(X, y)

    assert f1_score(y, ldep.predict(X)) >= 0.8
