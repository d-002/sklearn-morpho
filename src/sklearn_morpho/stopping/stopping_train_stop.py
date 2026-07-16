from . import StoppingMethod


class TrainStopStoppingMethod(StoppingMethod):
    """
    Stopping method that triggers whenever the training cost gets to 0, that is the
    perceptron found a global minimum and will no longer try to improve, rendering
    validation-related stopping methods unusable.
    """

    def requires_validation(self) -> bool:
        return False

    def should_stop(
        self, n_epochs: int, train_cost: float, validation_cost: float
    ) -> bool:
        return train_cost <= 0
