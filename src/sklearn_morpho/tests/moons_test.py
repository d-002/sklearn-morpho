import numpy as np
from typing import cast

from sklearn_morpho.classifiers.ldep import LDEP
from sklearn.datasets import make_moons
from sklearn.metrics import f1_score

def test_moons(runs=20, n_samples=100):
    dep = LDEP()

    for _ in range(runs):
        X, y = make_moons(n_samples=n_samples)
        X, y = cast(np.ndarray, X), cast(np.ndarray, y) # for pyright
        dep.fit(X, y)

        assert f1_score(y, dep.predict(X)) > .83
