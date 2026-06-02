from sklearn.utils.estimator_checks import check_estimator

from sklearn_morpho.classifiers.ldep import LDEP

def test_check_estimator():
    ldep = LDEP()
    assert check_estimator(ldep)
