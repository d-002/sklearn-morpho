import numpy as np

from sklearn_morpho import TempBinaryClassifier as TBC
from sklearn_morpho.datasets.gaussian_orthant import SampleData

def _make_dataset(noise: float) -> SampleData:
    mean = np.array([0, 0])
    deviation = np.array([5, 5])
    quadrant_bound = np.multiply(np.random.random(2) - .5, deviation) + mean
    return SampleData(mean, deviation, quadrant_bound, 500, noise)

def _get_prop_well_classified(sample_data: SampleData, classifier: TBC) -> float:
    predicted = classifier.predict(sample_data.X)
    return 1 - sum(predicted ^ sample_data.Y) / sample_data.X.size

def test_train_separable_dataset():
    sample_data = _make_dataset(0)

    classifier = TBC()
    classifier.fit(sample_data.X, sample_data.Y)

    # take care of floating point imprecisions by not using 1, even though
    # the perceptron should correctly classify everything
    assert classifier.get_fit_cost() == 0
    assert _get_prop_well_classified(sample_data, classifier) >= .9999

def test_train_noisy_dataset():
    for noise in (.01, .1, .2):
        sample_data = _make_dataset(noise)

        classifier = TBC()
        classifier.fit(sample_data.X, sample_data.Y)

        threshold = (1 - noise) * .83
        assert _get_prop_well_classified(sample_data, classifier) >= threshold
