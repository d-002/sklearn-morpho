from friendly_dataset import friendly_dataset
from sklearn.metrics import f1_score

from sklearn_morpho.weighting import DistSampleWeighting
from sklearn_morpho.stopping import StoppingMethod, EpochStoppingMethod
from sklearn_morpho import MorphoPerceptron

kinds = ['max', 'min']


def test_init():
    for kind in kinds:
        MorphoPerceptron(kind=kind)  # type: ignore


def test_train_noverif():
    X, y = friendly_dataset()

    for kind in kinds:
        perceptron = MorphoPerceptron(kind=kind)  # type: ignore
        perceptron.fit(X, y)


def test_train():
    X, y = friendly_dataset()

    for kind in kinds:
        for use_dccp_library in [False, True]:
            perceptron = MorphoPerceptron(
                kind=kind,
                use_dccp_library=use_dccp_library,  # type: ignore
            )
            perceptron.fit(X, y)

            assert f1_score(y, perceptron.predict(X)) >= 0.8


def test_train_custom_weighted_stopping():
    X, y = friendly_dataset()

    stopping_methods: list[StoppingMethod] = [EpochStoppingMethod()]
    for kind in kinds:
        perceptron = MorphoPerceptron(
            kind=kind,  # type: ignore
            weighting_method=DistSampleWeighting(),
            stopping_methods=stopping_methods,
        )

        perceptron.fit(X, y)

        assert f1_score(y, perceptron.predict(X)) >= 0.8


def test_train_with_penalty():
    X, y = friendly_dataset()

    for kind in kinds:
        perceptron = MorphoPerceptron(kind=kind, penalty=0.1)  # type: ignore
        perceptron.fit(X, y)

        assert f1_score(y, perceptron.predict(X)) >= 0.8


def test_train_dccp():
    X, y = friendly_dataset()

    for kind in kinds:
        perceptron = MorphoPerceptron(
            kind=kind,  # type: ignore
            use_dccp_library=True,
        )
        perceptron.fit(X, y)

        assert f1_score(y, perceptron.predict(X)) >= 0.8


def test_train_dccp_with_penalty():
    X, y = friendly_dataset()

    for kind in kinds:
        perceptron = MorphoPerceptron(
            kind=kind,  # type: ignore
            penalty=0.1,
            use_dccp_library=True,
        )
        perceptron.fit(X, y)

        assert f1_score(y, perceptron.predict(X)) >= 0.8
