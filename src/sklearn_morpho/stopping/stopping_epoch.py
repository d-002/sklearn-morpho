from . import StoppingMethod

class EpochStoppingMethod(StoppingMethod):
    """
    Stopping method that triggers whenever the number of epochs gets greater or
    equal to the parameter.
    """

    def __init__(self, max_epochs = 20):
        if max_epochs <= 0:
            raise ValueError('invalid max_epochs, expected > 0 but got '
                             f'{max_epochs}')

        self.max_epochs = max_epochs

    def requires_validation(self) -> bool:
        return False

    def should_stop(self, n_epochs: int, train_cost: float,
                    validation_cost: float) -> bool:
        return n_epochs >= self.max_epochs
