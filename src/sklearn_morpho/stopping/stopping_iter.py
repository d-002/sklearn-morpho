from .stopping_base import StoppingMethod

class IterStoppingMethod(StoppingMethod):
    """
    Stopping method that triggers whenever the number of iterations gets greater
    than the parameter.
    """

    def __init__(self, max_iterations: int):
        if max_iterations <= 0:
            raise ValueError('invalid max_iterations, expected > 0 but got '
                             f'{max_iterations}')

        self.max_iterations = max_iterations

    def requires_validation(self) -> bool:
        return False

    def should_stop(self, n_iterations: int, train_cost: float,
                    validation_cost: float) -> bool:
        return n_iterations > self.max_iterations
