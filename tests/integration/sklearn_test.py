from sklearn.utils.estimator_checks import check_estimator

from sklearn_morpho import DEP, LDEP, MorphoPerceptron
from sklearn_morpho.utils import Kind


def test_check_estimator_ldep() -> None:
    ldep = LDEP()
    assert check_estimator(ldep)


def test_check_estimator_dep() -> None:
    dep = DEP()
    assert check_estimator(dep)


def test_check_estimator_max_perceptron_str() -> None:
    perceptron = MorphoPerceptron(kind='max')
    assert check_estimator(perceptron)


def test_check_estimator_min_perceptron_str() -> None:
    perceptron = MorphoPerceptron(kind='min')
    assert check_estimator(perceptron)


def test_check_estimator_max_perceptron_enum() -> None:
    perceptron = MorphoPerceptron(kind=Kind.MAX)
    assert check_estimator(perceptron)


def test_check_estimator_min_perceptron_enum() -> None:
    perceptron = MorphoPerceptron(kind=Kind.MIN)
    assert check_estimator(perceptron)
