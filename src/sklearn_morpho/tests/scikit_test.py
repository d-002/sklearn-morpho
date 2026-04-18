from sklearn.utils.estimator_checks import check_estimator

from sklearn_morpho.classifiers.ldep import LDEP

def test_check_estimator():
    assert check_estimator(LDEP())
