from sklearn.utils.estimator_checks import check_estimator

from sklearn_morpho.classifiers.dep import DilationErosionPerceptron as DEP

def test_dummy():
    assert check_estimator(DEP())
