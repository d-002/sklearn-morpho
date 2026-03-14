from __future__ import annotations
import numpy as np
import cvxpy as cp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..perceptron import Perceptron

def bin_classify_total_cost(value: np.floating, y: np.ndarray) -> cp.Constraint | None:
    if y == 0:
        return np.max(np.array([value, 0]))
    elif y == 1:
        return np.max(np.array([-value, 0]))
    else:
        raise ValueError(f"Unknown class: {y}")

def compute_gradient(p: Perceptron, weights: np.ndarray, X: np.ndarray,
                     Y: np.ndarray, sample_radius: float) -> np.ndarray:
    """
    Approximate the gradient of a cost function at a specific point
    for all given data
    """

    N = p.dim
    gradient = np.empty(N)
    run = 2 * sample_radius

    for axis in range(N):
        diff = np.zeros(N)
        diff[axis] = sample_radius

        forward_1 = p.forward(weights - diff)
        forward_2 = p.forward(weights + diff)
        cost_1 = sum(bin_classify_total_cost(forward_1, y)
                     for _, y in zip(X, Y))
        cost_2 = sum(bin_classify_total_cost(forward_2, y)
                     for _, y in zip(X, Y))

        rise = cost_2 - cost_1
        gradient[axis] = rise / run

    return gradient

def train_gradient(p: Perceptron, X: np.ndarray, Y: np.ndarray,
                   max_iterations: int = 1000, done_threshold: float = .001,
                   max_gradient_norm: float = 10., learning_rate = .001,
                   learning_rate_decay: float = 0.99,
                   sample_radius: float = .01, verbose: bool = False) -> float:
    """
    Train a perceptron for data classification.
    Use stochastic gradient descent until the maximum number of iterations is
    reached, or the cost function does not change enough between iterations.

    return: the final cost, or -1 if no training happened.
    """

    if not X.shape[0]:
        return 0

    p.weights = np.zeros(p.dim)
    p.bias = p.get_neutral_bias()

    i = -1
    done = False
    prev_cost: float = np.inf
    cost: float = np.inf
    for i in range(max_iterations):
        #compute the gradient
        gradient = compute_gradient(
                p, p.weights, X, Y, sample_radius)
        # normalize the gradient if needed
        length = np.linalg.norm(gradient)
        if length > max_gradient_norm:
            gradient /= length / max_gradient_norm

        # compute the cost with the current perceptron weights
        cost = float(sum(bin_classify_total_cost(p.forward(x), y)
                         for x, y in zip(X, Y)))

        # break condition
        if abs(prev_cost - cost) < done_threshold:
            done = True
            break

        # move the perceptron's weights along the gradient
        p.weights -= gradient * learning_rate

        if verbose and (i + 1) % 100 == 0:
            print(f'Iteration {i + 1}, cost: {cost:.2f}, weights: {p.weights}')

        prev_cost = cost
        learning_rate *= learning_rate_decay

    if verbose:
        if done:
            print(f'Gradient descent done in {i} iterations, '
                  f'final cost is {cost:.2f}')
        else:
            print('Warning: reached max iterations for gradient descent')

    return cost
