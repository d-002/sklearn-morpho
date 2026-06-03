"""
Create and train different estimators and display their respective scores
averaged from multiple datasets.
Estimators selection inspired by arxiv/2011.06512
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
from time import time
from scipy.sparse._csr import csr_matrix
from joblib import Parallel, delayed
from multiprocessing import Manager

from sklearn.metrics import f1_score
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn_morpho import LDEP, RDEP, MorphoPerceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

## Configuration

FILE = 'comparison.json'
# ignore warnings for cleaner logs
ignore_warnings = True
n_folds = 5
max_jobs = 15
random_state = np.random.RandomState()

if ignore_warnings:
    warnings.filterwarnings("ignore")

print(f'Random state: {random_state}')
print(f'Comparison data will be outputted to: "{FILE}"')

## Set up datasets


def get_clean_openml(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    kwargs.setdefault('as_frame', False)
    kwargs.setdefault('version', 1)
    X, y = fetch_openml(name, return_X_y=True, **kwargs)

    # make sure X and y are not sparse
    if isinstance(X, csr_matrix):
        X = X.toarray()
    if isinstance(y, csr_matrix):
        y = y.toarray()

    # make sure X contains only numbers, not yes/no like in australian
    oe = OrdinalEncoder()
    X = oe.fit_transform(X)

    return X, y


skf = StratifiedKFold(n_splits=n_folds)
datasets_names = [
    'acute-inflammations',
    'australian',
    'banana',
    'banknote-authentication',
    'blood-transfusion-service-center',
    'breast-cancer',
    'colic',
    'credit-approval',
    'credit-g',
    'cylinder-bands',
    'diabetes',
    'eeg-eye-state',
    'haberman',
    'hill-valley',
    'ilpd',
    'ionosphere',
    'mofn-3-7-10',
    'monks-problems-2',
    'mushroom',
    'phoneme',
    'PhishingWebsites',
    'sick',
    'sonar',
    'spambase',
    'steel-plates-fault',
    'thoracic-surgery',
    'tic-tac-toe',
    'titanic',
]

# datasets not included compared to arxiv/2011.06512:
# - chess (nonexistent)
# - internet-advertisements (internally referring to a nonexistent dataset)

datasets_options = {
    'australian': {'version': 4},
    'Breast_Cancer_Wisconsin': {'as_frame': True},
    'cylinder-bands': {'version': 6},
    'titanic': {'as_frame': True},
}

## Create estimators

print('Creating estimators...')
estimators = {
    'l-DEP': OneVsRestClassifier(LDEP(random_state=random_state)),
    'DCCP l-DEP': OneVsRestClassifier(
        LDEP(use_dccp_library=True, random_state=random_state)
    ),
    'r-DEP': OneVsRestClassifier(RDEP(random_state=random_state)),
    'DCCP r-DEP': OneVsRestClassifier(
        RDEP(use_dccp_library=True, random_state=random_state)
    ),
    'Morpho_max': OneVsRestClassifier(
        MorphoPerceptron(kind='max', random_state=random_state)
    ),
    'Morpho_min': OneVsRestClassifier(
        MorphoPerceptron(kind='min', random_state=random_state)
    ),
    'Linear SVC': LinearSVC(random_state=random_state),
    'RBF SVC': SVC(kernel='rbf', random_state=random_state),
    'MLP': MLPClassifier(max_iter=1000, random_state=random_state),
    'Poly SVC': SVC(kernel='poly', random_state=random_state),
}

datasets_names = datasets_names[:5] # TODO remove

## Load datasets
print('\nLoading datasets...')

datasets = {}
for dataset_name in datasets_names:
    print(f'\033[32m=>\033[m Loading {dataset_name:<35}', end='\r')
    match dataset_name:
        case 'breast-cancer':
            datasets[dataset_name] = load_breast_cancer(return_X_y=True)
        case _:
            datasets[dataset_name] = get_clean_openml(
                dataset_name, **datasets_options.get(dataset_name, {})
            )
print('\n')

## Main logic

# centralized data
manager = Manager()
scores = manager.dict({d: {e: 0 for e in estimators} for d in datasets_names})
times = manager.dict({d: {e: 0 for e in estimators} for d in datasets_names})

def save_data():
    with open(FILE, 'w') as f:
        data = {'n_folds': n_folds, 'scores': scores, 'times': times}
        json.dump(data, f, indent=2)


def worker(scores: dict[str, dict[str, float]], times: dict[str, dict[str, float]], dataset_name: str, estimator_name: str) -> None:
    if ignore_warnings:
        warnings.filterwarnings("ignore")

    score_total = 0
    time_total = 0

    X, y = datasets[dataset_name]
    estimator = make_pipeline(
        SimpleImputer(strategy='mean'),  # remove NaNs
        estimators[estimator_name],
    )

    for i_train, i_test in skf.split(X, y):
        X_train, X_test = X[i_train], X[i_test]
        y_train, y_test = y[i_train], y[i_test]

        t0 = time()
        estimator.fit(X_train, y_train)
        t1 = time()

        score = f1_score(y_test, estimator.predict(X_test), average='micro')
        score_total += score
        time_total += t1 - t0

    # need to make copies because the variables are shared
    temp_scores = scores[dataset_name]
    temp_scores[estimator_name] = score_total / n_folds
    temp_times = times[dataset_name]
    temp_times[estimator_name] = time_total / n_folds

    scores[dataset_name] = temp_scores
    times[dataset_name] = temp_times


## train in parallel
print('Preparing to train')


n_jobs = min(os.process_cpu_count(), len(datasets), max_jobs)
if n_jobs <= 0:
    print('Nothing to do')
    sys.exit(0)

print(f'{len(datasets)} datasets to process, splitting work in {n_jobs} jobs.')

Parallel(n_jobs=n_jobs)(
    delayed(worker)(scores, times, dataset_name, estimator_name)
    for dataset_name in datasets
    for estimator_name in estimators
)

scores, times = dict(scores), dict(times)

print('Done.')
save_data()  # TODO remove
