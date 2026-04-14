import numpy as np

from sklearn_morpho.classifiers.dep import DilationErosionPerceptron as DEP
from sklearn.datasets import make_classification

def _get_prop_well_classified(X: np.ndarray, y: np.ndarray,
                              classifier: DEP) -> float:
    predicted = classifier.predict(X)
    return 1 - sum(predicted ^ y) / X.size

def test_train_separable_dataset():
    # use a random state to guarantee a separable dataset
    random_state = np.random.RandomState(11)
    X, y = make_classification(n_samples=50, n_features=2, n_redundant=0,
                               n_classes=2, n_clusters_per_class=1,
                               random_state=random_state)

    classifier = DEP(random_state=random_state)
    classifier.fit(X, y)

    assert np.allclose(_get_prop_well_classified(X, y, classifier), 1.)
