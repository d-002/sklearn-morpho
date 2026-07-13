from sklearn_morpho import stopping


def test_init() -> None:
    stopping.TrainStopStoppingMethod()


def test_validation() -> None:
    method = stopping.TrainStopStoppingMethod()
    assert not method.requires_validation()


def test_logic() -> None:
    method = stopping.TrainStopStoppingMethod()

    # also make sure the validation cost is not the one used
    assert not method.should_stop(0, 1, 1)
    assert not method.should_stop(0, 1, 0)
    assert method.should_stop(0, 0, 0)
    assert method.should_stop(0, 0, 1)
