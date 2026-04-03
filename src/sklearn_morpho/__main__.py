from .classifiers.dep import DilationErosionPerceptron as DEP

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.inspection import DecisionBoundaryDisplay

    from .datasets.gaussian import dataset_gaussians

    """
    Showcase of the features of this package.
    - Create random sample data
    - Create and train a perceptron with cvxpy and dccp
    #- Create and train another perceptron with naive gradient descent
    - Display and compare the results with matplotlib
    """

    # create sample data, assign colors
    method = 'wdccp'
    np.random.seed(8)
    sample_data = dataset_gaussians(50, 2, np.array(['red', 'blue']),
                                    np.random.rand(2, 2) * 10 - 5,
                                    (np.random.rand(2) * 2 + 1))
    X, y = sample_data

    # create and train perceptrons
    dep = DEP(method=method, margin=1, verbose=1)
    dep.fit(X, y)

    # compute and display perceptron decision region
    disp = DecisionBoundaryDisplay.from_estimator(
        dep, X, response_method='decision_function',
        grid_resolution=200,
        plot_method='contour',
        levels=[0],
        colors='black'
    )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')

    # show stats
    predicted = dep.predict(sample_data[0])
    accuracy = sum(int(y_predicted == y)
                   for y_predicted, y in zip(predicted, sample_data[1])) \
                           / len(sample_data[0])
    disp.ax_.title.set_text(f'l-DEP with WDCCP: cost {dep.fit_cost_:.2f}, '
                      f'accuracy {accuracy * 100:.2f}%')

    plt.show()
