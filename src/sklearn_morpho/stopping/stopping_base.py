from abc import ABC, abstractmethod

class StoppingMethod(ABC):
    """
    Abstract class for a fitting stopping method.
    """

    @abstractmethod
    def requires_validation(self) -> bool:
        """
        Whether the stopping method requires the training to be split in
        train/validation sets.
        """

    @abstractmethod
    def should_stop(self, n_iterations: int, train_cost: float,
                    validation_cost: float) -> bool:
        """
        Given some information, return whether the fitting process should stop.
        """
