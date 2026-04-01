import numpy as np

def dataset_gaussians(K: int, N: int, classes: np.ndarray, pos: np.ndarray,
                      deviation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample data generator for data classification.
    Creates labelled training data belonging to the given classes, distributed
    according to the normal distribution around the given poles.

    param K:       The total number of sample points, if not divisible by the
                   number of classes throw an exception.
    param N:       The size of each data point
    param classes: The classes to use for labelling
    pos:           An array of positions to use as center for the different
                   classes
    deviation:     An array of values to use for the standard deviation of each
                   class

    return:        A tuple [X, Y], where X is the set of randomly generated
                   points and Y is their associated labels
    """

    X = []
    Y = []

    k, rest = divmod(K, len(classes))
    if rest != 0:
        raise ValueError(f'Imbalanced number of samples ({K} over ' \
                         f'{len(classes)} classes) means biased dataset')

    for y, d, p in zip(classes, deviation, pos):
        points = np.multiply(np.random.randn(k, N), d) + p
        X += list(points)
        Y += [y.copy() for _ in range(k)]

    return np.array(X), np.array(Y)
