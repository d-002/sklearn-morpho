from .classifiers.dep import DilationErosionPerceptron as DEP

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    from .datasets.gaussian import dataset_gaussians
    from .training.dccp import get_wdccp_weights

    """
    Showcase of the features of this package.
    - Create random sample data
    - Create and train a perceptron with cvxpy and dccp
    #- Create and train another perceptron with naive gradient descent
    - Display and compare the results with matplotlib
    """

    # create sample data, assign colors
    sample_data = dataset_gaussians(500, 2, np.array(['red', 'blue']),
                                    np.array([[3, 2], [-4, -1]]),
                                    np.array([3, 1.5]))

    # create and train perceptrons
    dep = DEP(verbose=True)
    dep.fit(*sample_data)

    # display and compare results with matplotlib:
    fig, ax = plt.subplots()
    names = ('Weighted DCCP with CvxPy', 'DCCP with CvxPy',
             'Naive gradient descent')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # show stats
    predicted = dep.predict(sample_data[0])
    accuracy = sum(int(y_predicted == y)
                   for y_predicted, y in zip(predicted, sample_data[1])) \
                           / len(sample_data[0])
    ax.title.set_text(f'DEP with WDCCP: cost {dep.fit_cost_:.2f}, '
                      f'accuracy {accuracy * 100:.2f}%')

    # display wdccp weights as points transparency, if applicable
    if dep.method == 'wdccp':
        classes = list(set(sample_data[1]))
        y_integers = np.array([classes.index(y) for y in sample_data[1]])
        wdccp_weights, _ = get_wdccp_weights(sample_data[0], y_integers)
    else:
        wdccp_weights = np.ones(sample_data[0].shape[0])

    # display target sample_data classification
    for x, y, w in zip(*sample_data, wdccp_weights):
        ax.scatter(*x, color=y, alpha=np.sin(w * np.pi / 2))

    # compute and display perceptron decision region
    ((A, u), (B, v)) = dep.get_decision_region_points()
    ax.add_patch(PathPatch(Path([A + 20 * u, A, B, B + 20 * v],
                                [Path.MOVETO, Path.LINETO,
                                 Path.LINETO, Path.LINETO]),
                           fill=None))

    plt.show()
