"""
Create and train different estimators and display their respective scores
averaged from multiple datasets.
Estimators selection inspired by arxiv/2011.06512
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import make_classification, make_moons, \
        load_breast_cancer

from sklearn_morpho.classifiers.ldep import LDEP

# set up estimators and datasets
random_state = np.random.RandomState(11)
estimators = {
    'l_DEP': LDEP(),
    'RBF SVC': SVC(kernel='rbf'),
    'Poly SVC': SVC(kernel='poly'),
    'Linear SVC': SVC(kernel='linear'),
    'MLP': MLPClassifier(),
}

datasets = {
    'normal classification': make_classification(
        n_samples=500, n_features=2, n_redundant=0, n_classes=2,
        n_clusters_per_class=1, random_state=random_state),
    'breast cancer': load_breast_cancer(return_X_y=True),
    'non noisy moons': make_moons(n_samples=500, random_state=random_state),
    'noisy moons': make_moons(n_samples=500, noise=.2,
                              random_state=random_state),
}

# evaluate estimators
scores = {estimator_name: [] for estimator_name in estimators}
times = {estimator_name: [] for estimator_name in estimators}

skf = StratifiedKFold(n_splits=5)
for dataset_name, (X, y) in datasets.items():
    print(f"Training with dataset '{dataset_name}'...")
    pos_label = np.unique(y)[1]

    for estimator_name, estimator in estimators.items():
        print(f'  - Estimator {estimator_name}...')

        for X_fold, y_fold in skf.split(X, y):
            X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=.5)

            t0 = time()
            estimator.fit(X_train, y_train)
            t1 = time()

            score = f1_score(y_train, estimator.predict(X_train),
                             pos_label=pos_label)
            scores[estimator_name].append(score)
            times[estimator_name].append(t1 - t0)
print('Done.')

# display results
scores = {name: np.array(score) for name, score in scores.items()}
times = {name: np.array(time) for name, time in times.items()}

fig, axs = plt.subplots(ncols=2, nrows=1)
# log scale for times
axs[1].set_yscale('log')

for data, name, ax in zip((scores, times),
                          ('average F1 score', 'Training time (s)'), axs):
    ax.set_title(name)
    ax.boxplot(data.values(), patch_artist=True, tick_labels=data.keys())

plt.show()
