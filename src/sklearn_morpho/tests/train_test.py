import numpy as np

from sklearn_morpho.classifiers.ldep import LDEP
from sklearn.datasets import make_classification

def test_train_separable_dataset():
    # use a random state to guarantee a separable dataset
    random_state = np.random.RandomState(11)
    X, y = make_classification(n_samples=50, n_features=2, n_redundant=0,
                               n_classes=2, n_clusters_per_class=1,
                               random_state=random_state)

    classifier = LDEP(random_state=random_state)
    classifier.fit(X, y)

    fails = X.shape[0] - np.sum(classifier.predict(X) == y)
    assert fails == 0
