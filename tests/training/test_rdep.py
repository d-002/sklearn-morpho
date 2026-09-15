import numpy as np
import pytest
from friendly_dataset import friendly_dataset
from sklearn.metrics import f1_score

from sklearn_morpho import RDEP
from sklearn.svm import SVC
from sklearn_morpho.inversion import NoInversion
from sklearn_morpho.stopping import EpochStoppingMethod, StoppingMethod
from sklearn_morpho.training import SOLVER_DCCP
from sklearn_morpho.weighting import DistSampleWeighting


def test_init() -> None:
    RDEP()


def test_train_noverif() -> None:
    dep = RDEP()

    X, y = friendly_dataset()
    dep.fit(X, y)


def test_train_degenerate_dataset() -> None:
    dep = RDEP()

    X, y = np.zeros((2, 1)), np.arange(2)
    with pytest.raises(ValueError):
        dep.fit(X, y)


def test_train() -> None:
    X, y = friendly_dataset()

    for solver in [None, SOLVER_DCCP]:
        dep = RDEP(solver=solver)
        dep.fit(X, y)

        assert f1_score(y, dep.predict(X)) >= 0.8


def test_train_override_preprocessing() -> None:
    X, y = friendly_dataset()

    estimators = [SVC()]

    for solver in [None, SOLVER_DCCP]:
        dep = RDEP(estimators, solver=solver)
        dep.fit(X, y)

        assert f1_score(y, dep.predict(X)) >= 0.8
