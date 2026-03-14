from sklearn.utils.estimator_checks import check_estimator

from sklearn_morpho.binary import MorphologicalClassifier

def test_dummy():
    assert check_estimator(MorphologicalClassifier())
