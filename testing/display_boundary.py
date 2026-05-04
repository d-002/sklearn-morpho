import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, cast
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_classification, make_moons

from sklearn_morpho.classifiers.ldep import LDEP

"""
Create and train a perceptron with cvxpy and DCCP for multiple datasets,
display the results and show the decision region if possible for each of them
"""

# create sample data, assign colors
# 1
random_state = np.random.RandomState(11)

datasets = {
    'normal classification': make_classification(
        n_samples=500, n_features=2, n_redundant=0, n_classes=2,
        n_clusters_per_class=1, random_state=random_state),
    'non noisy moons': make_moons(n_samples=500, random_state=random_state),
    'noisy moons': make_moons(n_samples=500, noise=.2,
                              random_state=random_state),
}

total_test_score = 0
for name, (X, y) in datasets.items():
    print(f'Training with "{name}" dataset...')

    y = np.array(['red', 'blue'])[y]
    pos_label = np.unique(y)[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

    # for LSPs
    X_train, X_test = cast(np.ndarray, X_train), cast(np.ndarray, X_test)
    y_train, y_test = cast(np.ndarray, y_train), cast(np.ndarray, y_test)

    # create and train estimator
    dep = LDEP(margin=1, verbose=1, random_state=random_state)
    dep.fit(X_train, y_train)
    score_train = f1_score(y_train, dep.predict(X_train),
                           pos_label=pos_label)
    score_test = f1_score(y_test, dep.predict(X_test), pos_label=pos_label)

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
        ax.title.set_text(f'l-DEP: F1 score {score_test * 100:.3f}%')
        plt.show()

    print(f'F1 score on training set: {score_train * 100:.3f}%, '
          f'on testing set: {score_test * 100:.3f}%')

    total_test_score += score_test

n_datasets = len(datasets)
print(f'Done with all {n_datasets} datasets, average test score: '
      f'{total_test_score / n_datasets * 100:.3f}')
