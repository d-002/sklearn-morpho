from abc import ABC, abstractmethod
from typing import Literal, cast
import numpy as np
import cvxpy as cp
from time import time

from ..weighting.weighting_base import SampleWeighting

class DccpTrainer(ABC):
    """
    Abstract class for optimization using cvxpy
    """

    def __init__(self, weighting_method: SampleWeighting, margin: float,
                 max_iterations: int, done_threshold: float,
                 verbose: Literal[0, 1, 2],
                 random_state: np.random.RandomState) -> None:
        """
        Initialize the trainer.

        param weighting_method: The weighting method to use: apply weights to
                                the cost contribution of each data point to help
                                avoid outliers.
        param max_iterations:   Upper bound for the number of training
                                iterations.
        param done_threshold:   If the cost changes by a number smaller than
                                this value in between operations, or itself goes
                                below this value, stop early.
        param verbose:          Whether to log extra information.
        """

        if margin < 0:
            raise ValueError('invalid margin, expected >= 0 but got {margin}')
        if max_iterations <= 0:
            raise ValueError('invalid max_iterations, expected > 0 but got '
                             f'{max_iterations}')
        if done_threshold <= 0:
            raise ValueError('invalid done_threshold, expected > 0 but got '
                             f'{done_threshold}')

        self.weighting_method = weighting_method
        self.margin = margin
        self.max_iterations = max_iterations
        self.done_threshold = done_threshold
        self.verbose = verbose
        self.random_state = random_state

    def at_training_start(self) -> None:
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

        start = time()
        wdccp_weights, cost_normalizer = \
                self.weighting_method.fit_transform(X, y)

        i = -1
        done = False
        prev_cost: float = np.inf
        cost: float = -1

        # formulate the cvxpy problem to solve, common to all iterations
        self.at_training_start()

        for i in range(self.max_iterations):
            problem = self.get_problem(X, y, wdccp_weights)

            # solve the problem, normalize the cost when using wdccp
            cost = cast(float, problem.solve(verbose=self.verbose == 2)) \
                    * cost_normalizer
            cost_adjustment = abs(cost - prev_cost)

            self.after_iteration()

            # logging and loop logic
            if self.verbose:
                print(f'Iteration {i + 1}/{self.max_iterations}, '
                      f'cost: {cost:.8f}, adjustment: {cost_adjustment}')

            if min(cost, cost_adjustment) <= self.done_threshold:
                done = True
                break
            prev_cost = cost

        if self.verbose:
            dt = time() - start
            if done:
                print(f'DCCP done in {i + 1}/{self.max_iterations} iterations, '
                      f'final cost is {cost:.8f} in {dt:.2f}s')
            else:
                print('Warning: reached max iterations for DCCP after '
                        f'{dt:.2f}s')

        self.at_training_end()

        return cost
