from .classifiers.dep import DilationErosionPerceptron as DEP

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import cast
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import DecisionBoundaryDisplay
    from sklearn.datasets import make_moons, make_classification

    from .datasets.wdbc import dataset_wdbc

    """
    Showcase of the features of this package.
    - Create random sample data or load a dataset
    - Create and train a perceptron with cvxpy and DCCP
    - Display the results
    """

    # create sample data, assign colors
    random_state = None
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                               n_classes=2, n_clusters_per_class=1,
                               random_state=random_state)
    #X, y = dataset_wdbc('WDBC.dat.txt')
    #X, y = make_moons(n_samples=500)
    y = np.array(['red', 'blue'])[y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

    # for LSPs
    X_train, X_test = cast(np.ndarray, X_train), cast(np.ndarray, X_test)
    y_train, y_test = cast(np.ndarray, y_train), cast(np.ndarray, y_test)

    # create and train estimator
    dep = DEP(method='dccp', margin=1, verbose=1, random_state=random_state)
    dep.fit(X_train, y_train)
    accuracy_train = np.sum(dep.predict(X_train) == y_train) / len(X_train)
    accuracy_test = np.sum(dep.predict(X_test) == y_test) / len(X_test)

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
        ax.title.set_text(f'l-DEP with {dep.method}: cost {dep.fit_cost_:.8f}, '
                          f'accuracy {accuracy_test * 100:.3f}%')
        plt.show()

    print(f'Accuracy on training set: {accuracy_train * 100:.3f}%, '
          f'testing set: {accuracy_test * 100:.3f}%')
