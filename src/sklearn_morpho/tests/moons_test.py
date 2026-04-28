import numpy as np

from sklearn_morpho.classifiers.ldep import LDEP
from sklearn.datasets import make_moons

def test_moons(n_samples=500):
    # use a random state to guarantee a separable dataset
    random_state = np.random.RandomState(11)
    dep = LDEP(margin=1, random_state=random_state)

    X, y = make_moons(n_samples=n_samples, random_state=random_state)
    dep.fit(X, y)

    fails = X.shape[0] - np.sum(dep.predict(X) == y)
    assert fails == 0
