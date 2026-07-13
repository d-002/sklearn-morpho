from typing import cast

import numpy as np
import pytest
from friendly_dataset import friendly_dataset

from sklearn_morpho import LDEP
from sklearn_morpho.stopping import (
    EpochStoppingMethod,
    StoppingMethod,
    TrainStopStoppingMethod,
)
from sklearn_morpho.training.dccp_ldep import LDEPDccpTrainer
from sklearn_morpho.weighting import NoneSampleWeighting


def test_init() -> None:
    dims = (10, 10)
    none = NoneSampleWeighting()
    stopping_methods: list[StoppingMethod] = [EpochStoppingMethod(1)]

    with pytest.raises(ValueError):
        LDEPDccpTrainer(
            dims,
            -1,
            0,
            0,
            none,
            stopping_methods,
            False,
            0,
            np.random.RandomState(),
        )
    with pytest.raises(ValueError):
        LDEPDccpTrainer(
            dims,
            0,
            -1,
            0,
            none,
            stopping_methods,
            False,
            0,
            np.random.RandomState(),
        )
    with pytest.raises(ValueError):
        LDEPDccpTrainer(
            dims,
            0,
            0,
            -1,
            none,
            stopping_methods,
            False,
            0,
            np.random.RandomState(),
        )
    with pytest.raises(ValueError):
        LDEPDccpTrainer(
            dims,
            0,
            0,
            1,
            none,
            stopping_methods,
            False,
            0,
            np.random.RandomState(),
        )
    with pytest.raises(ValueError):
        LDEPDccpTrainer(
            dims,
            0,
            0,
            2,
            none,
            stopping_methods,
            False,
            0,
            np.random.RandomState(),
        )
    with pytest.raises(ValueError):
        LDEPDccpTrainer(
            dims, 0, 0, 0, none, [], False, 0, np.random.RandomState()
        )


def test_train_no_validation() -> None:
    stopping_methods = [TrainStopStoppingMethod(), EpochStoppingMethod(10)]

    ldep = LDEP(validation_ratio=0, stopping_methods=stopping_methods)
    ldep_failing = LDEP(validation_ratio=0)

    X, y = friendly_dataset()

    ldep.fit(X, y)

    with pytest.raises(ValueError):
        ldep_failing.fit(X, y)


def test_train_verbose_1_no_validation() -> None:
    stopping_methods = [TrainStopStoppingMethod(), EpochStoppingMethod(10)]

    ldep = LDEP(
        validation_ratio=0, stopping_methods=stopping_methods, verbose=1
    )

    X, y = friendly_dataset()

    ldep.fit(X, y)


def test_train_verbose_1() -> None:
    ldep = LDEP(verbose=1)
    X, y = friendly_dataset()
    ldep.fit(X, y)


def test_train_verbose_2() -> None:
    ldep = LDEP(verbose=2)
    X, y = friendly_dataset()
    ldep.fit(X, y)


def test_train_dccp_verbose_1() -> None:
    ldep = LDEP(use_dccp_library=True, verbose=1)
    X, y = friendly_dataset()
    ldep.fit(X, y)


def test_train_dccp_verbose_2() -> None:
    ldep = LDEP(use_dccp_library=True, verbose=2)
    X, y = friendly_dataset()
    ldep.fit(X, y)
