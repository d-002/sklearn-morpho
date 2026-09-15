import numpy as np
import pytest
from friendly_dataset import friendly_dataset
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from sklearn_morpho import RDEP
from sklearn_morpho.inversion import NoInversion
from sklearn_morpho.stopping import EpochStoppingMethod, StoppingMethod
from sklearn_morpho.training import SOLVER_DCCP
from sklearn_morpho.weighting import DistSampleWeighting


def test_init() -> None:
    RDEP()


def test_train_noverif() -> None:
    rdep = RDEP()

    X, y = friendly_dataset()
    rdep.fit(X, y)


def test_train_degenerate_dataset() -> None:
    rdep = RDEP()

    X, y = np.zeros((2, 1)), np.arange(2)
    with pytest.raises(ValueError):
        rdep.fit(X, y)


def test_train_no_estimators() -> None:
    rdep = RDEP([])

    X, y = friendly_dataset()
    with pytest.raises(ValueError):
        rdep.fit(X, y)


def test_train() -> None:
    X, y = friendly_dataset()

    for solver in [None, SOLVER_DCCP]:
        rdep = RDEP(solver=solver)
        rdep.fit(X, y)

        assert f1_score(y, rdep.predict(X)) >= 0.8


def test_train_override_preprocessing() -> None:
    X, y = friendly_dataset()

    estimators = [SVC()]

    for solver in [None, SOLVER_DCCP]:
        rdep = RDEP(estimators, solver=solver)
        rdep.fit(X, y)

        assert f1_score(y, rdep.predict(X)) >= 0.8
