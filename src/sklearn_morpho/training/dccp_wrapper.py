from abc import ABC, abstractmethod
from typing import Literal, cast
import numpy as np
import cvxpy as cp
from time import time

def get_wdccp_weights(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Get weights for the WDCCP method: for a given class, all its elements will
    receive a weight that will determine how they impact the final cost,
    decreasing linearly as their distance to their class centroid increases,
    from 1 at the centroid to 0 for the farthest point of the class.

    param X: The set of data points
    param y: The respective classes

    return:  A tuple containing both the weights and a normalizing factor, to be
             multiplied with the finally calculated cost so that it would be
             comparable in value to one calculated without WDCCP weights.
    """

    K = X.shape[0]

    # compute centroids, there should be no classes with no elements thanks to
    # sklearn checks
    labels, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    sums = np.zeros((len(labels), X.shape[1]))
    np.add.at(sums, inv, X)
    centroids = sums / counts[:, np.newaxis]

    # inverse distance from each data point to its respective class centroid
    wdccp_weights = 1 / (1e-6 + np.linalg.norm(X - centroids[y], axis=1))
    max_centroid_w = np.array([wdccp_weights[y == y_].max() for y_ in range(2)])
    wdccp_weights /= max_centroid_w[y]

    # since the cost can be lower than 1, compensate final cost calculation
    cost_normalizer = wdccp_weights.sum()
    cost_normalizer = 1 if cost_normalizer == 0 else K / cost_normalizer

    return wdccp_weights, cost_normalizer

class DccpTrainer(ABC):
    """
    Abstract class for optimization using cvxpy
    """

    def __init__(self, weighted: bool, margin: float, max_iterations: int,
                 batch_size: int, done_threshold: float,
                 verbose: Literal[0, 1, 2],
                 random_state: np.random.RandomState) -> None:
        """
        Initialize the trainer.

        param weighted:       Whether to use WDCCP: apply weights to the cost
                              contribution of each data point depending on how
                              far they are from the class's centroid, so that
                              outliers contribute less to the final cost.
        param max_iterations: Upper bound for the number of training iterations.
        param batch_size:     Batch size for mini batch fitting, or zero for
                              no batching.
        param done_threshold: If the cost changes by a number smaller than this
                              value in between operations, or itself goes below
                              this value, stop early.
        param verbose:        Whether to log extra information.
        """

        if margin < 0:
            raise ValueError('invalid margin, expected >= 0 but got {margin}')
        if max_iterations <= 0:
            raise ValueError('invalid max_iterations, expected > 0 but got '
                             f'{max_iterations}')
        if batch_size < 0:
            raise ValueError('invalid batch_size, expected >= 0 but got '
                             f'{batch_size}')
        if done_threshold <= 0:
            raise ValueError('invalid done_threshold, expected > 0 but got '
                             f'{done_threshold}')

        self.weighted = weighted
        self.margin = margin
        self.max_iterations = max_iterations
        self.batch_size = batch_size
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
                    wdccp_weights: np.ndarray
                    ) -> tuple[cp.Minimize | cp.Maximize, list[cp.Constraint]]:
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
        K = X.shape[0]

        # for WDCCP, compute the centroid of each one of the two regions
        if self.weighted:
            wdccp_weights, cost_normalizer = get_wdccp_weights(X, y)
        else:
            # put all weights to 1 to ignore them but maintain code readability
            wdccp_weights, cost_normalizer = np.ones(K), 1

        i = -1
        done = False
        prev_cost: float = np.inf
        cost: float = -1

        # formulate the cvxpy problem to solve, common to all iterations
        self.at_training_start()

        for i in range(self.max_iterations):
            objective, constraints = self.get_problem(X, y, wdccp_weights)

            # solve the problem, normalize the cost when using wdccp
            prob = cp.Problem(objective, constraints)
            cost = cast(float, prob.solve(verbose=self.verbose != 2)) \
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
                print(f'{'W' if self.weighted else ''}DCCP done in '
                      f'{i + 1}/{self.max_iterations} iterations, '
                      f'final cost is {cost:.8f} in {dt:.2f}s')
            else:
                print('Warning: reached max iterations for DCCP after '
                        f'{dt:.2f}s')

        self.at_training_end()

        return cost
