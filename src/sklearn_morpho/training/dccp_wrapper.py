from abc import ABC, abstractmethod
from typing import Any
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
                 margin: float, max_iterations: int, done_threshold: float,
                 verbose: bool) -> None:
        """
        Initialize the trainer.

        param perceptrons:    A list of perceptrons that will be trained later.
        param weighted:       Whether to use WDCCP: apply weights to the cost
                              contribution of each data point depending on how
                              far they are from the class's centroid, so that
                              outliers contribute less to the final cost.
        param max_iterations: Upper bound for the number of training iterations.
        param done_threshold: If the cost changes by a number smaller than this
                              value in between operations, or itself goes below
                              this value, stop early.
        param verbose:        Whether to log extra information.
        """

        if margin < 0:
            raise ValueError('margin is too low, expected >= 0 but got '
                             f'{margin}')
        if max_iterations <= 0:
            raise ValueError('max_iterations is too low, expected > 0 but got '
                             f'{max_iterations}')
        if done_threshold <= 0:
            raise ValueError('done_threshold is too low, expected > 0 but got '
                             f'{done_threshold}')

        self.perceptrons = perceptrons
        self.weighted = weighted
        self.margin = margin
        self.max_iterations = max_iterations
        self.done_threshold = done_threshold
        self.verbose = verbose

    def at_training_start(self) -> list[cp.Constraint]:
        """
        Optional actions to take when the training starts.
        For example, create and persist a list of additional variables to
        optimize.

        Return a list of additional constraints, or an empty list if not
        applicable.

        If this method is overriden, the derived class must call this base
        method to initialize the perceptrons.
        """

        # will be populated during training, but need an initial value for
        # linearization
        for perceptron in self.perceptrons:
            perceptron.weights = np.zeros(perceptron.dim)

        return []

    def after_training_iteration(self, optimized_weights: list[cp.Variable]
                                 ) -> None:
        """
        Optional actions to take when the training ends.
        """

    def at_training_end(self) -> None:
        """
        Optional actions to take when the training is over.
        """

    @abstractmethod
    def cvx_cost_function(self, weights: list[cp.Variable], x: np.ndarray,
                          y: Any, slack: cp.Variable,
                          k: int) -> cp.Constraint | None:
        """
        Compute the convex part of the full cost function for a specific input,
        then optionally return a constraint.

        param weights: The weights that are currently being optimized
        param x:       The current data point
        param y:       The class of that particular data point
        param slack:   The slack variable to use in the constraint
        param k:       The index inside that slack variable

        return:        A constraint, or None if there is no constraint for this
                       particular data point.
        """

    @abstractmethod
    def ccv_cost_function_made_convex(self, weights: list[cp.Variable],
                                      x: np.ndarray, y: Any, slack: cp.Variable,
                                      k: int) -> cp.Constraint | None:
        """
        Compute the concave part of the full cost function for a specific input,
        then optionally return a constraint.
        The calculated concave cost function must also be convex, i.e. it has to
        be linearized, for example by calculating the plane tangent to the cost
        function evaluated at x.

        See self.cvx_cost_function for similar parameters and return value.
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
        K = X.shape[0]
        additional_constraints = self.at_training_start()

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
        slack = cp.Variable(K)
        optimized_weights = [cp.Variable(perceptron.dim)
                             for perceptron in self.perceptrons]
        objective = cp.Minimize(
                cp.sum(cp.multiply(cp.pos(slack), wdccp_weights)))

        for i in range(self.max_iterations):
            constraints = additional_constraints[:]

            for k, (x, y) in enumerate(zip(X, Y)):
                # add the convex constraints
                cvx_constraint = self.cvx_cost_function(
                        optimized_weights, x, y, slack, k)
                if cvx_constraint is not None:
                    constraints.append(cvx_constraint)

                # add the linearized concave constraints
                ccv_constraint = self.ccv_cost_function_made_convex(
                        optimized_weights, x, y, slack, k)
                if ccv_constraint is not None:
                    constraints.append(ccv_constraint)

            # solve the problem, normalize the cost when using wdccp
            prob = cp.Problem(objective, constraints)
            cost = prob.solve() * cost_normalizer
            cost_adjustment = abs(cost - prev_cost)

            if self.verbose:
                print(f'Iteration {i + 1}/{self.max_iterations}, '
                      f'cost: {cost:.2f}, adjustment: {cost_adjustment}')

            # update the weights early for argmax/argmin sampling
            self.after_training_iteration(optimized_weights)

            if min(cost, cost_adjustment) <= self.done_threshold:
                done = True
                break
            prev_cost = cost

        if self.verbose:
            dt = time() - start
            if done:
                print(f'{'W' if self.weighted else ''}DCCP done in '
                      f'{i}/{self.max_iterations} iterations, '
                      f'final cost is {cost:.2f} in {dt:.2f}s')
            else:
                print('Warning: reached max iterations for DCCP after '
                        f'{dt:.2f}s')

        self.at_training_end()

        return cost
