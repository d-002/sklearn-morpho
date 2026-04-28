import numpy as np

from sklearn_morpho.classifiers.ldep import LDEP
from sklearn.datasets import make_moons

def test_separable_moons(runs=10):
    random_state = None
    ok = 0
    dep = LDEP(margin=1, random_state=random_state)

    for _ in range(runs):
        X, y = make_moons(n_samples=500)
        dep.fit(X, y)
        dep.max_iterations = 50

        fails = X.shape[0] - np.sum(dep.predict(X) == y)
        ok += fails == 0

    # very slow, will need to get back to this
    # basically sometimes the matrices get really big, but adding their norm to
    # the cost makes things converge slower and iterations veeery expensive
    assert ok / runs >= .5
