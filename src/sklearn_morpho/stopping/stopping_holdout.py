from .stopping_base import StoppingMethod

class HoldoutStoppingMethod(StoppingMethod):
    """
    Stopping method that triggers whenever the validation cost starts increasing
    while the training cost keeps decreasing, this for a set number of
    consecutive iterations.
    """

    def __init__(self, delay: int):
        if delay <= 0:
            raise ValueError(f'invalid delay, expected > 0 but got {delay}')

        self.delay = delay
        self.count = 0

        self.prev_train_cost: float | None = None
        self.prev_validation_cost: float | None = None

    def should_stop(self, n_iterations: int, train_cost: float,
                    validation_cost: float) -> bool:
        if self.prev_train_cost is None or self.prev_validation_cost is None:
            self.prev_train_cost = train_cost
            self.prev_validation_cost = validation_cost
        else:
            if train_cost < self.prev_train_cost and \
                    validation_cost > self.prev_validation_cost:
                self.count += 1
                if self.count >= self.delay:
                    return True
            else:
                self.count = 0

        return False
