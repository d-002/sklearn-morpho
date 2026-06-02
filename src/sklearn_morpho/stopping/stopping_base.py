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

        Can still be False even if using validation_cost.
        In fact, the should_stop method should use the validation cost as the
        main and most reliable cost, in case of no validation it will just be
        the same as the training cost.
        """

    @abstractmethod
    def should_stop(self, n_epochs: int, train_cost: float,
                    validation_cost: float) -> bool:
        """
        Given some information, return whether the fitting process should stop.
        """
