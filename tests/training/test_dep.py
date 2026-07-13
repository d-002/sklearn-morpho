import numpy as np
import pytest
from friendly_dataset import friendly_dataset
from sklearn.metrics import f1_score

from sklearn_morpho import DEP
from sklearn_morpho.inversion import NoInversion
from sklearn_morpho.stopping import EpochStoppingMethod, StoppingMethod
from sklearn_morpho.weighting import DistSampleWeighting


def test_init() -> None:
    DEP()


@pytest.mark.filterwarnings('error')
def test_train_invalid_params() -> None:
    X, y = friendly_dataset()

    DEP()
    DEP(lambda_bounds=(0, 1)).fit(X, y)

    with pytest.raises(ValueError):
        DEP(lambda_bounds=(-1, 1)).fit(X, y)
    with pytest.raises(ValueError):
        DEP(lambda_bounds=(0, 1.1)).fit(X, y)
    with pytest.raises(ValueError):
        DEP(lambda_bounds=(-1, 2)).fit(X, y)

    with pytest.raises(UserWarning):
        DEP(lambda_bounds=(0, 1), use_dccp_library=True).fit(X, y)
    with pytest.raises(ValueError):
        DEP(lambda_bounds=(-1, 2), use_dccp_library=True).fit(X, y)


def test_train_noverif() -> None:
    dep = DEP()

    X, y = friendly_dataset()
    dep.fit(X, y)


def test_train_degenerate_dataset() -> None:
    dep = DEP()

    X, y = np.zeros((2, 1)), np.arange(2)
    with pytest.raises(ValueError):
        dep.fit(X, y)


def test_train() -> None:
    X, y = friendly_dataset()

    for use_dccp_library in [False, True]:
        dep = DEP(use_dccp_library=use_dccp_library)
        dep.fit(X, y)

        assert f1_score(y, dep.predict(X)) >= 0.8


def test_train_custom_weighted_stopping_invert() -> None:
    stopping_methods: list[StoppingMethod] = [EpochStoppingMethod()]
    dep = DEP(
        weighting_method=DistSampleWeighting(),
        stopping_methods=stopping_methods,
        inversion_method=NoInversion(),
    )

    X, y = friendly_dataset()
    dep.fit(X, y)

    assert f1_score(y, dep.predict(X)) >= 0.8


def test_train_with_penalty() -> None:
    dep = DEP(penalty=0.1)

    X, y = friendly_dataset()
    dep.fit(X, y)

    assert f1_score(y, dep.predict(X)) >= 0.8


def test_train_dccp() -> None:
    dep = DEP(use_dccp_library=True)

    X, y = friendly_dataset()
    dep.fit(X, y)

    assert f1_score(y, dep.predict(X)) >= 0.8


def test_train_dccp_with_penalty() -> None:
    dep = DEP(penalty=0.1, use_dccp_library=True)

    X, y = friendly_dataset()
    dep.fit(X, y)

    assert f1_score(y, dep.predict(X)) >= 0.8
