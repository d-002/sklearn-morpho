from abc import ABC, abstractmethod
from typing import Literal, cast
import numpy as np
import cvxpy as cp
from time import time

from sklearn.model_selection import train_test_split
from ..weighting.weighting_base import SampleWeighting
from ..stopping.stopping_base import StoppingMethod

class DccpTrainer(ABC):
    """
    Abstract class for DCCP optimization using cvxpy.
    """

    def __init__(self, margin: float, validation_ratio: float,
                 weighting_method: SampleWeighting,
                 stopping_methods: list[StoppingMethod],
                 verbose: Literal[0, 1, 2],
                 random_state: np.random.RandomState) -> None:
        """
        Initialize the trainer.

        param validation_radio: How much of the training set to dedicate to use
                                as validation during fitting.
        param weighting_method: The weighting method to use: apply weights to
                                the cost contribution of each data point to help
                                avoid outliers.
        param stopping_methods: A list of stopping methods, must not be empty.
                                At each iteration, these methods will be
                                sequentially asked whether the training should
                                stop. In this case, training ends by rolling
                                back to the iteration with the best validation
                                cost.
        param verbose:          Whether to log extra information. 0: no logging,
                                1: basic logging / timing, 2: cvxpy solve() set
                                to verbose mode.
        param random_state:     A RandomState object or None to allow for seeded
                                randomness.
        """

        if margin < 0:
            raise ValueError(f'Invalid margin, expected >= 0 but got {margin}')
        if not 0 < validation_ratio < 1:
            raise ValueError('Invalid validation ratio, expected > 0 and < 1 '
                             f'but got {validation_ratio}')
        if not len(stopping_methods):
            raise ValueError('Empty list of stopping methods, training would '
                             'run indefinitely')

        self.margin = margin
        self.validation_ratio = validation_ratio
        self.weighting_method = weighting_method
        self.stopping_methods = stopping_methods
        self.verbose = verbose
        self.random_state = random_state

    def at_training_start(self, data_dim: int) -> None:
        """
        Optional actions to take when the training starts.
        For example, create and persist a list of additional variables to
        optimize.
        """

    def after_iteration(self) -> None:
        """
        Optional additional actions to take after each training iteration.
        """

    def at_training_end(self) -> None:
        """
        Optional actions to take when the training is over.
        """

    @abstractmethod
    def get_problem(self, X: np.ndarray, y: np.ndarray,
                    wdccp_weights: np.ndarray) -> cp.Problem:
        """
        Compute a cvxpy Objective and a list of Constraints for use in DCCP.
        The prlblem must be DCP, meaning the constraints must all be convex.
        For concave constraints, they must be linearized beforehand, possibly
        from constant values taken from the previous iteration result.

        param X:             The data points
        param y:             The data points classes
        param wdccp_weights: The wdccp weights, all 1 if not using wdccp, for
                             use in computing the objective.
                             This is an array with one element corresponding to
                             an element in X.

        return:              A tuple containing a cvxpy Objective and a list of
                             Constraints.
        """

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Train using DCCP.
        This means this function can only solve problems where the cost function
        can be separable in the given convex and concave parts.

        param X: The set of data points to use for training.
        param y: The set of labels associated with elements of X.

        return:  The final cost, or -1 if no training happened.
        """

        if self.verbose:
            print('Starting fitting with DCCP')

        X_train, X_validation, y_train, y_validation = train_test_split(
                X, y, test_size=self.validation_ratio)
        X_train = cast(np.ndarray, X_train)
        X_validation = cast(np.ndarray, X_validation)
        y_train = cast(np.ndarray, y_train)
        y_validation = cast(np.ndarray, y_validation)

        if not X_train.size or not X_validation.size:
            raise ValueError('Current validation ratio makes degenerate '
                             'train/validation split: '
                             + str(self.validation_ratio))

        start = time()
        wdccp_weights, cost_normalizer = \
                self.weighting_method.fit_transform(X_train, y_train)

        iteration = 1
        cost: float = -1

        # formulate the cvxpy problem to solve, common to all iterations
        self.at_training_start(X_train.shape[1])

        while True:
            problem = self.get_problem(X_train, y_train, wdccp_weights)

            # solve the problem, normalize the cost when using wdccp
            cost = cast(float, problem.solve(verbose=self.verbose == 2)) \
                    * cost_normalizer

            self.after_iteration()

            # logging and loop logic
            if self.verbose:
                print(f'Iteration {iteration}, cost: {cost:.8f}')

            done = False
            validation_cost = 0 # TODO
            for stopping_method in self.stopping_methods:
                if stopping_method.should_stop(
                        iteration, cost, validation_cost):
                    done = True
                    break

            if done:
                break

            iteration += 1

        if self.verbose:
            dt = time() - start
            print(f'DCCP done in {iteration} iterations, '
                  f'final cost is {cost:.8f} in {dt:.2f}s')

        self.at_training_end()

        return cost
