"""
Create and train different estimators and display their respective scores
averaged from multiple datasets.
Estimators selection inspired by arxiv/2011.06512
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Literal, cast
from time import time
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import make_classification, make_moons, \
        load_breast_cancer

from sklearn_morpho.classifiers.ldep import LDEP

# helper classes because different classifiers have different needs depending
# on the dataset
class EstimatorWrapper(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def setup(self, dataset_name: str) -> BaseEstimator:
        pass

class LDEPWrapper(EstimatorWrapper):
    methods: dict[str, Literal['dccp', 'wdccp']] = {
        'normal classification': 'wdccp',
        'breast cancer': 'wdccp',
        'non noisy moons': 'dccp',
        'noisy moons': 'dccp',
    }

    def __init__(self) -> None:
        super().__init__('l_DEP')

    def setup(self, dataset_name: str) -> BaseEstimator:
        return LDEP(method=self.methods[dataset_name], margin=1)

# set up estimators and datasets
random_state = np.random.RandomState(11)
estimators: list[EstimatorWrapper] = [LDEPWrapper()]

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
scores = {estimator_wrapper.name: [] for estimator_wrapper in estimators}
times = {estimator_wrapper.name: [] for estimator_wrapper in estimators}

skf = StratifiedKFold(n_splits=5)
for dataset_name, (X, y) in datasets.items():
    print(f"Training with dataset '{dataset_name}'...")
    pos_label = np.unique(y)[1]

    for estimator_wrapper in estimators:
        print(f'  - Estimator {estimator_wrapper.name}...')
        estimator = estimator_wrapper.setup(dataset_name)

        for X_fold, y_fold in skf.split(X, y):
            X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=.5)

            t0 = time()
            estimator.fit(X_train, y_train)
            t1 = time()

            score = f1_score(y_train, estimator.predict(X_train),
                             pos_label=pos_label)
            scores[estimator_wrapper.name].append(score)
            times[estimator_wrapper.name].append(t1 - t0)
print('Done.')

# display results
scores = {name: np.array(score) for name, score in scores.items()}
times = {name: np.array(time) for name, time in times.items()}

fig, axs = plt.subplots(ncols=2, nrows=1)

for data, name, ax in zip((scores, times),
                          ('average F1 score', 'Training time (s)'), axs):
    ax.set_title(name)
    ax.boxplot(data.values(), patch_artist=True, tick_labels=data.keys())

plt.show()
