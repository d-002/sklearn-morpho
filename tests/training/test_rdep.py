import pytest
import numpy as np

from friendly_dataset import friendly_dataset
from sklearn.metrics import f1_score

from sklearn_morpho.weighting import DistSampleWeighting
from sklearn_morpho.stopping import StoppingMethod, EpochStoppingMethod
from sklearn_morpho import RDEP

def test_init():
    RDEP()

@pytest.mark.filterwarnings("error")
def test_train_invalid_params():
    X, y = friendly_dataset()

    RDEP()
    RDEP(lambda_bounds=(0, 1)).fit(X, y)

    with pytest.raises(ValueError):
        RDEP(lambda_bounds=(-1, 1)).fit(X, y)
    with pytest.raises(ValueError):
        RDEP(lambda_bounds=(0, 1.1)).fit(X, y)
    with pytest.raises(ValueError):
        RDEP(lambda_bounds=(-1, 2)).fit(X, y)

    with pytest.raises(UserWarning):
        RDEP(lambda_bounds=(0, 1), use_dccp_library=True).fit(X, y)
    with pytest.raises(ValueError):
        RDEP(lambda_bounds=(-1, 2), use_dccp_library=True).fit(X, y)

def test_train_noverif():
    rdep = RDEP()

    X, y = friendly_dataset()
    rdep.fit(X, y)

def test_train_degenerate_dataset():
    rdep = RDEP()

    X, y = np.zeros((2, 1)), np.arange(2)
    with pytest.raises(ValueError):
        rdep.fit(X, y)

def test_train():
    rdep = RDEP()

    X, y = friendly_dataset()
    rdep.fit(X, y)

    assert f1_score(y, rdep.predict(X)) >= 0.8

def test_train_custom_weighted_stopping():
    stopping_methods: list[StoppingMethod] = [EpochStoppingMethod()]
    rdep = RDEP(weighting_method=DistSampleWeighting(),
                stopping_methods=stopping_methods)

    X, y = friendly_dataset()
    rdep.fit(X, y)

    assert f1_score(y, rdep.predict(X)) >= 0.8

def test_train_with_penalty():
    rdep = RDEP(penalty=.1)

    X, y = friendly_dataset()
    rdep.fit(X, y)

    assert f1_score(y, rdep.predict(X)) >= 0.8

def test_train_dccp():
    rdep = RDEP(use_dccp_library=True)

    X, y = friendly_dataset()
    rdep.fit(X, y)

    assert f1_score(y, rdep.predict(X)) >= 0.8

def test_train_dccp_with_penalty():
    rdep = RDEP(penalty=.1, use_dccp_library=True)

    X, y = friendly_dataset()
    rdep.fit(X, y)

    assert f1_score(y, rdep.predict(X)) >= 0.8
