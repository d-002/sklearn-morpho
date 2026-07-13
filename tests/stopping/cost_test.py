import pytest

from sklearn_morpho import stopping


def test_init() -> None:
    stopping.CostStoppingMethod(1)

    with pytest.raises(ValueError):
        stopping.CostStoppingMethod(0)
    with pytest.raises(ValueError):
        stopping.CostStoppingMethod(-1)


def test_validation() -> None:
    method = stopping.CostStoppingMethod(1)
    assert not method.requires_validation()


def test_logic() -> None:
    for threshold in (0.1, 1, 10, 1e6):
        method = stopping.CostStoppingMethod(threshold)

        assert method.should_stop(0, 0, threshold * 0.99)
        # make sure the training cost is not taken into account
        assert method.should_stop(0, threshold * 2, threshold * 0.99)

        assert method.should_stop(0, threshold + 1, threshold)
        assert method.should_stop(0, 0, threshold)

        assert not method.should_stop(0, threshold + 1, threshold + 1)
        assert not method.should_stop(0, 0, threshold + 1)
