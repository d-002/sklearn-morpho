import pytest

from sklearn_morpho import stopping

def test_init():
    stopping.EarlyStoppingMethod(1)

    with pytest.raises(ValueError):
        stopping.EarlyStoppingMethod(0)
    with pytest.raises(ValueError):
        stopping.EarlyStoppingMethod(-1)

def test_validation():
    method = stopping.EarlyStoppingMethod(1)
    assert method.requires_validation()

def test_stop_at_start():
    for delay in (1, 5, 100):
        method = stopping.EarlyStoppingMethod(delay)

        for i in range(delay):
            assert not method.should_stop(0, 0, i)

        assert method.should_stop(0, 0, delay)

def test_reset_then_stop():
    method = stopping.EarlyStoppingMethod(3)

    assert not method.should_stop(0, 0, 1)
    assert not method.should_stop(0, 0, 2)
    assert not method.should_stop(0, 0, 3)

    assert not method.should_stop(0, 0, 0) # new best score, reset count
    assert not method.should_stop(0, 0, 1)
    assert not method.should_stop(0, 0, 2)

    assert method.should_stop(0, 0, 1)
