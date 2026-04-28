from abc import ABC, abstractmethod

class StoppingMethod(ABC):
    """
    Abstract class for a fitting stopping method.
    """

    @abstractmethod
    def should_stop(self, n_iterations: int, train_cost: float,
                    validation_cost: float) -> bool:
        """
        Given some information, return whether the fitting process should stop.
        """
