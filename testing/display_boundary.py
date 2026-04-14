from sklearn_morpho.classifiers.dep import DilationErosionPerceptron as DEP

"""
Create and train a perceptron with cvxpy and DCCP for multiple datasets,
display the results and show the decision region if possible for each of them
"""

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from typing import Literal, cast
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import DecisionBoundaryDisplay
    from sklearn.datasets import make_classification, make_moons, \
            load_breast_cancer

    # create sample data, assign colors
    random_state = np.random.RandomState(11)

    datasets = {
        'normal classification': make_classification(
            n_samples=500, n_features=2, n_redundant=0, n_classes=2,
            n_clusters_per_class=1, random_state=random_state),
        'breast cancer': load_breast_cancer(return_X_y=True),
        'non noisy moons': make_moons(n_samples=500),
        'noisy moons': make_moons(n_samples=500, noise=.2),
    }
    methods: dict[str, Literal['dccp', 'wdccp']] = {
        'normal classification': 'wdccp',
        'breast cancer': 'wdccp',
        'non noisy moons': 'dccp',
        'noisy moons': 'dccp',
    }

    total_test_accuracy = 0
    for name, (X, y) in datasets.items():
        print(f'Training with "{name}" dataset...')
        method = methods[name]

        print(np.unique(y))
        y = np.array(['red', 'blue'])[y]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

        # for LSPs
        X_train, X_test = cast(np.ndarray, X_train), cast(np.ndarray, X_test)
        y_train, y_test = cast(np.ndarray, y_train), cast(np.ndarray, y_test)

        # create and train estimator
        dep = DEP(method=method, margin=1, verbose=1, random_state=random_state)
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
            ax.scatter(*X_train.T, color=y_train, alpha=.2)
            ax.scatter(*X_test.T, color=y_test)
            ax.title.set_text(f'l-DEP with {method}: '
                              f'training cost {dep.fit_cost_:.8f}, '
                              f'test accuracy {accuracy_test * 100:.3f}%')
            plt.show()

        print(f'Accuracy on training set: {accuracy_train * 100:.3f}%, '
              f'testing set: {accuracy_test * 100:.3f}%')

        total_test_accuracy += accuracy_test

    n_datasets = len(datasets)
    print(f'Done with all {n_datasets} datasets, average test accuracy: '
          f'{total_test_accuracy / n_datasets * 100:.3f}')
