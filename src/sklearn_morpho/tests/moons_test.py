import numpy as np

from sklearn_morpho.classifiers.ldep import LDEP
from sklearn.datasets import make_moons

from sklearn_morpho.stopping import CostStoppingMethod, IterStoppingMethod

def test_moons(runs=10, n_samples=500):
    dep = LDEP(validation_ratio=0, stopping_methods=[
        CostStoppingMethod(1e-6),
        IterStoppingMethod(100),
    ])

    for _ in range(runs):
        X, y = make_moons(n_samples=n_samples)
        dep.fit(X, y)

        fails = X.shape[0] - np.sum(dep.predict(X) == y)
        assert fails == 0

    # TODO will most likely fail the test, will get back to this
    # basically sometimes the matrices get really big, but adding their norm to
    # the cost makes things converge slower and iterations very expensive
