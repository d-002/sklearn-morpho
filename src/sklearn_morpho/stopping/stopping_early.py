from . import StoppingMethod


class EarlyStoppingMethod(StoppingMethod):
    """
Stopping method that triggers whenever the validation cost starts increasing
while the training cost keeps decreasing, this for a set number of consecutive
epochs.

It is assumed that the training cost keeps decreasing as should be the case
during functional training.
    """

    def __init__(self, delay: int = 5) -> None:
        if delay <= 0:
            raise ValueError(f'invalid delay, expected > 0 but got {delay}')

        self.delay = delay
        self.count = 0

        self.best_validation_cost: float | None = None

    def requires_validation(self) -> bool:
        return True

    def should_stop(
        self, n_epochs: int, train_cost: float, validation_cost: float
    ) -> bool:
        if self.best_validation_cost is None:
            self.best_validation_cost = validation_cost
        else:
            if validation_cost >= self.best_validation_cost:
                self.count += 1
                if self.count >= self.delay:
                    return True
            else:
                self.count = 0
                self.best_validation_cost = validation_cost

        return False
