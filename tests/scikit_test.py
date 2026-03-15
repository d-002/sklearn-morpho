from sklearn.utils.estimator_checks import check_estimator

from sklearn_morpho.classifiers.dep import DilationErosionPerceptron as DEP

def test_check_estimator():
    assert check_estimator(DEP(.5))
