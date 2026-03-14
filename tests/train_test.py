import numpy as np

from sklearn_morpho.classifiers.dep import DilationErosionPerceptron as DEP
from sklearn_morpho.datasets.gaussian import dataset_gaussians

def _get_prop_well_classified(X: np.ndarray, Y: np.ndarray,
                              classifier: DEP) -> float:
    predicted = classifier.predict(X)
    return 1 - sum(predicted ^ Y) / X.size

def test_train_separable_dataset():
    sample_data = dataset_gaussians(50, 2, np.array([0, 1]),
                                    np.array([[-3, 2], [4, -1]]),
                                    np.array([3, 1.5]))

    classifier = DEP()
    classifier.fit(*sample_data)

    # take care of floating point imprecisions by not using 1, even though
    # the perceptron should correctly classify everything
    assert _get_prop_well_classified(*sample_data, classifier) >= .9999
