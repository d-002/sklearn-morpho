import pytest

from sklearn_morpho import stopping


def test_init() -> None:
    stopping.EpochStoppingMethod(1)

    with pytest.raises(ValueError):
        stopping.EpochStoppingMethod(0)
    with pytest.raises(ValueError):
        stopping.EpochStoppingMethod(-1)


def test_validation() -> None:
    method = stopping.EpochStoppingMethod(1)
    assert not method.requires_validation()


def test_logic() -> None:
    for epochs in (1, 5, 100):
        method = stopping.EpochStoppingMethod(epochs)

        for i in range(epochs):
            assert not method.should_stop(i, 0, 0)

        assert method.should_stop(epochs, 0, 0)
        assert method.should_stop(epochs + 1, 0, 0)
