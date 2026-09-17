from friendly_dataset import friendly_dataset
from sklearn.metrics import f1_score

from sklearn_morpho import MorphoPerceptron
from sklearn_morpho.inversion import NoInversion
from sklearn_morpho.stopping import EpochStoppingMethod, StoppingMethod
from sklearn_morpho.training import SOLVER_DCCP
from sklearn_morpho.utils import Kind
from sklearn_morpho.weighting import DistSampleWeighting


def test_init() -> None:
    for kind in Kind:
        MorphoPerceptron(kind=kind)


def test_train_noverif() -> None:
    X, y = friendly_dataset()

    for kind in Kind:
        perceptron = MorphoPerceptron(kind=kind)
        perceptron.fit(X, y)


def test_train() -> None:
    X, y = friendly_dataset()

    for kind in Kind:
        for solver in [None, SOLVER_DCCP]:
            perceptron = MorphoPerceptron(
                kind=kind,
                solver=solver,
            )
            perceptron.fit(X, y)

            assert f1_score(y, perceptron.predict(X)) >= 0.8


def test_train_custom_weighted_stopping_invert() -> None:
    X, y = friendly_dataset()

    stopping_methods: list[StoppingMethod] = [EpochStoppingMethod()]
    for kind in Kind:
        perceptron = MorphoPerceptron(
            kind=kind,
            weighting_method=DistSampleWeighting(),
            stopping_methods=stopping_methods,
            inversion_method=NoInversion(),
        )

        perceptron.fit(X, y)

        assert f1_score(y, perceptron.predict(X)) >= 0.8


def test_train_with_penalty() -> None:
    X, y = friendly_dataset()

    for kind in Kind:
        perceptron = MorphoPerceptron(kind=kind, penalty=0.1)
        perceptron.fit(X, y)

        assert f1_score(y, perceptron.predict(X)) >= 0.8


def test_train_dccp() -> None:
    X, y = friendly_dataset()

    for kind in Kind:
        perceptron = MorphoPerceptron(
            kind=kind,
            solver=SOLVER_DCCP,
        )
        perceptron.fit(X, y)

        assert f1_score(y, perceptron.predict(X)) >= 0.8


def test_train_dccp_with_penalty() -> None:
    X, y = friendly_dataset()

    for kind in Kind:
        perceptron = MorphoPerceptron(
            kind=kind,
            penalty=0.1,
            solver=SOLVER_DCCP,
        )
        perceptron.fit(X, y)

        assert f1_score(y, perceptron.predict(X)) >= 0.8
