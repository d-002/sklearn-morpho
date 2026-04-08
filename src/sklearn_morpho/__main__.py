from .classifiers.dep import DilationErosionPerceptron as DEP

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import cast
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import DecisionBoundaryDisplay

    from .datasets.gaussian import dataset_gaussians
    from .datasets.wdbc import dataset_wdbc

    """
    Showcase of the features of this package.
    - Create random sample data or load a dataset
    - Create and train a perceptron with cvxpy and DCCP
    - Display the results
    """

    # create sample data, assign colors
    method = 'wdccp'
    X, y = dataset_gaussians(500, 2, np.array(['red', 'blue']),
                                    np.random.rand(2, 2) * 10 - 5,
                                    (np.random.rand(2) * 2 + 1))
    #X, y = dataset_wdbc('WDBC.dat.txt')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

    # for LSPs
    X_train, X_test = cast(np.ndarray, X_train), cast(np.ndarray, X_test)
    y_train, y_test = cast(np.ndarray, y_train), cast(np.ndarray, y_test)

    # create and train perceptrons
    dep = DEP(method=method, margin=1, verbose=1)
    dep.fit(X_train, y_train)

    # compute and display perceptron decision region
    if X.shape[1] == 2:
        disp = DecisionBoundaryDisplay.from_estimator(
            dep, X_test, response_method='decision_function',
            grid_resolution=200,
            plot_method='contour',
            levels=[0],
            colors='black'
        )
        ax = disp.ax_
        ax.scatter(*X_test.T, color=y_test)
    else:
        _, ax = plt.subplots()

    # show stats
    accuracy_train = np.sum(dep.predict(X_train) == y_train) / len(X_train)
    accuracy_test = np.sum(dep.predict(X_test) == y_test) / len(X_test)
    ax.title.set_text(f'l-DEP with WDCCP: cost {dep.fit_cost_:.8f}, '
                      f'accuracy {accuracy_test * 100:.3f}%')
    print(f'Accuracy on training set: {accuracy_train * 100:.3f}%, '
          f'testing set: {accuracy_test * 100:.3f}%')

    plt.show()
