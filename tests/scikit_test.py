from sklearn.utils.estimator_checks import check_estimator

from sklearn_morpho import TempBinaryClassifier as TBC

def test_dummy():
    assert check_estimator(TBC())
