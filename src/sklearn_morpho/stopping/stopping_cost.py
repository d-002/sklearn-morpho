from . import StoppingMethod

class CostStoppingMethod(StoppingMethod):
    """
    Stopping method that triggers whenever the cost gets lower or equal to the
    parameter.
    """

    def __init__(self, cost_threshold: float):
        if cost_threshold <= 0:
            raise ValueError('invalid done_threshold, expected > 0 but got '
                             f'{cost_threshold}')

        self.cost_threshold = cost_threshold

    def requires_validation(self) -> bool:
        return False

    def should_stop(self, n_iterations: int, train_cost: float,
                    validation_cost: float) -> bool:
        return validation_cost <= self.cost_threshold
