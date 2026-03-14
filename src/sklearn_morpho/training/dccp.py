from __future__ import annotations
import numpy as np
import cvxpy as cp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..perceptron import MaxPerceptron

def bin_classify_cvx_cost(k: int, value: cp.Variable, slack: cp.Variable,
             y: np.ndarray) -> cp.Constraint | None:
    if y == 0:
        return value <= slack[k]
    return None

def bin_classify_ccv_cost(k: int, value: cp.Variable, slack: cp.Variable,
             y: np.ndarray) -> cp.Constraint | None:
    if y == 1:
        return value >= -slack[k]
    return None

def get_wdccp_weights(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
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

def train_dccp(p: MaxPerceptron, X: np.ndarray, Y: np.ndarray,
               weighted: bool, max_iterations: int, done_threshold = 0.001,
               verbose: bool = False) -> float:
    """
    Train a perceptron for data classification.
    Use DCCP: the cost function is the sum of the positive terms of a slack
    obtained from a convex and concave part.
    Approximate the concave part into a hyperplane at the weights for each
    iteration.
    Can stop early if the cost does not change enough between iterations.

    param weight: whether to apply weights for each individual cost, so that
                  outliers contribute less to the final cost.

    return: the final cost, or -1 if no training happened.
    """

    N = p.dim
    K = X.shape[0]
    p.weights = np.zeros(N)
    p.bias = p.get_neutral_bias()

    if not K:
        return 0

    # for WDCCP, compute the centroid of each one of the two regions
    if weighted:
        wdccp_weights, cost_normalizer = get_wdccp_weights(X, Y)
    else:
        # put all weights to 1 to ignore them but maintain code readability
        wdccp_weights = np.ones(K)
        cost_normalizer = 1

    i = -1
    done = False
    prev_cost: float = np.inf
    cost: float = np.inf
    for i in range(max_iterations):
        # formulate the cvxpy problem to solve
        slack = cp.Variable(K)
        optimized_weights = cp.Variable(N)
        objective = cp.Minimize(
                cp.sum(cp.multiply(cp.pos(slack), wdccp_weights)))
        constraints = []

        for k, (x, y) in enumerate(zip(X, Y)):
            # add the convex constraints
            value = cp.max(optimized_weights + x)
            cvx_constraint = bin_classify_cvx_cost(k, value, slack, y)
            if cvx_constraint is not None:
                constraints.append(cvx_constraint)

            # add the concave constraints:
            # convexify the concave cost function by approximating it
            # since we are working with a max perceptron, the approximation can
            # be done by only using the weigth - input pair that maximizes the
            # cost function for the given data point
            index = np.argmax(p.weights + x)
            value = optimized_weights[index] + x[index]
            ccv_constraint = bin_classify_ccv_cost(k, value, slack, y)
            if ccv_constraint is not None:
                constraints.append(ccv_constraint)

        # solve the problem
        prob = cp.Problem(objective, constraints)

        # update and normalize the cose in case using wdccp
        cost = prob.solve() * cost_normalizer

        if abs(cost - prev_cost) < done_threshold:
            done = True
            break

        if verbose:
            print(f'Iteration {i + 1}/{max_iterations}, cost: {cost:.2f}, '
                  f'weights: {optimized_weights.value}')

        # update the weights for sampling
        p.weights = np.array(optimized_weights.value)
        prev_cost = cost

    if verbose:
        if done:
            print(f'{'W' if weighted else ''}DCCP done in {i} iterations, '
                  f'final cost is {cost:.2f}')
        else:
            print('Warning: reached max iterations for gradient descent')

    return cost
