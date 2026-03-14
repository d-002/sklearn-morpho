from . import MorphologicalClassifier as MC

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    from .binary.sample_data import SampleData
    from .binary.training.dccp import get_wdccp_weights

    """
    Showcase of the features of this package.
    - Create random sample data
    - Create and train a perceptron with cvxpy and dccp
    #- Create and train another perceptron with naive gradient descent
    - Display and compare the results with matplotlib
    """

    # create sample data, assign colors
    mean = np.array([0, 0])
    deviation = np.array([5, 5])
    quadrant_bound = np.multiply(np.random.random(2) - .5, deviation) + mean
    sample_data = SampleData(mean, deviation, quadrant_bound, 500, .05)
    colorize = lambda class_: 'red' if class_ else 'blue'

    # create and train perceptrons
    p_wdccp = MC('wdccp', verbose=True)
    p_dccp = MC('dccp', verbose=True)
    p_gradient = MC('gradient', verbose=True)
    p_wdccp.fit(sample_data.X, sample_data.Y)
    p_dccp.fit(sample_data.X, sample_data.Y)
    p_gradient.fit(sample_data.X, sample_data.Y)

    # display and compare results with matplotlib:
    fig, axs = plt.subplots(ncols=2, nrows=2)
    names = ('Weighted DCCP with CvxPy', 'DCCP with CvxPy',
             'Naive gradient descent')
    for p, name, ax in zip((p_wdccp, p_dccp, p_gradient), names, axs.flatten()):
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        # show stats
        predicted = p.predict(sample_data.X)
        accuracy = sum(int(y_predicted == y)
                       for y_predicted, y in zip(predicted, sample_data.Y)) \
                               / len(sample_data.X)
        ax.title.set_text(f'{name}: cost {p.get_fit_cost():.2f}, '
                          f'accuracy {accuracy * 100:.2f}%')

        if p.method == 'wdccp':
            wdccp_weights, _ = get_wdccp_weights(sample_data.X, sample_data.Y)
        else:
            wdccp_weights = np.ones(sample_data.X.shape[0])

        # display target sample_data classification
        for x, y, w in zip(sample_data.X, sample_data.Y, wdccp_weights):
            ax.scatter(*x, color=colorize(y), alpha=np.sin(w * np.pi / 2))

        # compute and display perceptron decision region
        x, y = -p.perceptron_.weights
        ax.add_patch(PathPatch(Path([(-10, y), (x, y), (x, -10)],
                                    [Path.MOVETO, Path.LINETO, Path.LINETO]),
                               fill=None))

    plt.show()
