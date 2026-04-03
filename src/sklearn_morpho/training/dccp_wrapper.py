from abc import ABC, abstractmethod
from typing import Any, Literal, cast
import numpy as np
import cvxpy as cp
from time import time

from ..perceptron import Perceptron

def get_wdccp_weights(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Get weights for the WDCCP method: for a given class, all its elements will
    receive a weight that will determine how they impact the final cost,
    decreasing linearly as their distance to their class centroid increases,
    from 1 at the centroid to 0 for the farthest point of the class.

    param X: The set of data points
    param Y: The respective classes

    return:  A tuple containing both the weights and a normalizing factor, to be
             multiplied with the finally calculated cost so that it would be
             comparable in value to one calculated without WDCCP weights.
    """

    N = X[0].size
    K = X.shape[0]

    centroids = np.zeros((2, N))
    counts = np.zeros(2)
    for x, y in zip(X, Y):
        centroids[y] += x
        counts[y] += 1
    centroids = np.array([np.zeros(N) if not count else centroid / count
                          for centroid, count in zip(centroids, counts)])

    # inverse distance from each data point to its respective class centroid
    temp_w = np.array([np.linalg.norm(x - centroids[y])
                       for x, y in zip(X, Y)])
    temp_w = np.array([0 if not w else 1 / w for w in temp_w])

    # normalization step to put all weights between 0 and 1 for each class
    max_centroid_w = np.array([max(d for y_, d in zip(Y, temp_w)
                                   if y_ == y)
                               for y in range(2)])
    wdccp_weights = np.array([d / max_centroid_w[y]
                              for y, d in zip(Y, temp_w)])

    # since the cost can be lower than 1, compensate final cost calculation
    cost_normalizer = sum(wdccp_weights)
    cost_normalizer = 1 if cost_normalizer == 0 else K / cost_normalizer

    return wdccp_weights, cost_normalizer

class DccpTrainer(ABC):
    """
    Abstract class for defining a trainer for a list of perceptrons using DCCP.
    """

    def __init__(self, perceptrons: list[Perceptron], weighted: bool,
                 margin: float, max_iterations: int, batch_size: int,
                 done_threshold: float, verbose: Literal[0, 1, 2],
                 rs: np.random.RandomState) -> None:
        """
        Initialize the trainer.

        param perceptrons:    A list of perceptrons that will be trained later.
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

        self.perceptrons = perceptrons
        self.weighted = weighted
        self.margin = margin
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.done_threshold = done_threshold
        self.verbose = verbose
        self.rs = rs

    def at_training_start(self) -> None:
        """
        Optional actions to take when the training starts.
        For example, create and persist a list of additional variables to
        optimize.

        If this method is overriden, the derived class must call this base
        method to initialize the perceptrons.
        """

        # will be populated during training, but need an initial value for
        # linearization
        for perceptron in self.perceptrons:
            perceptron.weights = np.random.random(perceptron.dim) * 2 - 1

    def after_iteration(self) -> None:
        """
        Optional additional actions to take after each training iteration.
        Before calling this method, the perceptron weights will have been
        updated.
        """

    def at_training_end(self) -> None:
        """
        Optional actions to take when the training is over.
        """

    @abstractmethod
    def get_problem(self, weights: list[cp.Variable], X: np.ndarray,
                    Y: np.ndarray, wdccp_weights: np.ndarray) -> tuple[
                            cp.Minimize | cp.Maximize, list[cp.Constraint]]:
        """
        Compute a cvxpy Objective and a list of Constraints for use in DCCP.
        The prlblem must be DCP, meaning the constraints must all be convex.
        For concave constraints, they must be linearized beforehand, possibly
        from constant values taken from the previous iteration result.

        param weights:       The weights that are currently being optimized, as
                             a list of arrays for each perceptron.
        param X:             The data points
        param Y:             The data points classes
        param wdccp_weights: The wdccp weights, all 1 if not using wdccp, for
                             use in computing the objective.
                             This is an array with one element corresponding to
                             an element in X.

        return:              A tuple containing a cvxpy Objective and a list of
                             Constraints.
        """

    def train(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Train a list of perceptrons using DCCP.
        This means this function can only solve problems where the cost function
        can be separable in the given convex and concave parts.

        param X: The set of data points to use for training.
        param Y: The set of labels associated with elements of X.

        return:  The final cost, or -1 if no training happened.
        """

        start = time()
        K, N = X.shape[0], self.perceptrons[0].dim

        # for WDCCP, compute the centroid of each one of the two regions
        if self.weighted:
            wdccp_weights, cost_normalizer = get_wdccp_weights(X, Y)
        else:
            # put all weights to 1 to ignore them but maintain code readability
            wdccp_weights, cost_normalizer = np.ones(K), 1

        i = -1
        done = False
        prev_cost: float = np.inf
        cost: float = -1

        # formulate the cvxpy problem to solve, common to all iterations
        weights = [cp.Variable(N) for _ in range(len(self.perceptrons))]
        self.at_training_start()

        for i in range(self.max_iterations):
            objective, constraints = self.get_problem(weights, X, Y,
                                                      wdccp_weights)

            # solve the problem, normalize the cost when using wdccp
            prob = cp.Problem(objective, constraints)
            cost = cast(float, prob.solve(verbose=self.verbose == 2)) \
                    * cost_normalizer
            cost_adjustment = abs(cost - prev_cost)

            # update the perceptrons weights from this iteration's results
            for perceptron, w in zip(self.perceptrons, weights):
                if w.value is None:
                    raise ValueError('CvxPy could not optimize a perceptron')
                perceptron.weights = w.value
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
                      f'{i}/{self.max_iterations} iterations, '
                      f'final cost is {cost:.8f} in {dt:.2f}s')
            else:
                print('Warning: reached max iterations for DCCP after '
                        f'{dt:.2f}s')

        self.at_training_end()

        return cost
