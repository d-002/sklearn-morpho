"""
Create and train different estimators and display their respective scores
averaged from multiple datasets.
Estimators selection inspired by arxiv/2011.06512
"""

import os
import sys
import json
import signal
import numpy as np
from time import time
from scipy.sparse._csr import csr_matrix
from multiprocessing import Manager, Pool

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

## Configuration, timeout handling

FILE = 'comparison_data.json'
n_folds = 5
timeout = 60

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException('Training timed out')


# TODO handle this
signal.signal(signal.SIGALRM, timeout_handler)

## set up estimators
random_state = np.random.RandomState()
print(f'Random state: {random_state}')

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
datasets = [
    'acute-inflammations',
    'australian',
    'banana',
    'banknote-authentication',
    'blood-transfusion-service-center',
    'breast-cancer',
    'chess',
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
# - internet-advertisements (internally referring to a nonexistent dataset)

datasets_options = {
    'australian': {'version': 4},
    'Breast_Cancer_Wisconsin': {'as_frame': True},
    'chess': {'version': 2},
    'cylinder-bands': {'version': 6},
    'titanic': {'as_frame': True},
}


## Main logic

# TODO add progress bar, etc
# TODO use save_data better
def save_data(scores, times):
    with open(FILE, 'w') as f:
        json.dump(
            {'n_folds': n_folds, 'scores': scores, 'times': times},
            f,
            indent=2,
        )


def worker(scores, times, dataset_name) -> None:
    # some datasets must be loaded differently
    match dataset_name:
        case 'breast-cancer':
            X, y = load_breast_cancer(return_X_y=True)
        case _:
            X, y = get_clean_openml(
                dataset_name, **datasets_options.get(dataset_name, {})
            )

    for estimator_name, estimator in estimators.items():
        print(f'{dataset_name:>35}', estimator_name) # TODO
        score_total = 0
        time_total = 0

        estimator = make_pipeline(
            SimpleImputer(strategy='mean'),  # remove NaNs
            estimator,
        )

        for i_train, i_test in skf.split(X, y):
            X_train, X_test = X[i_train], X[i_test]
            y_train, y_test = y[i_train], y[i_test]

            #signal.alarm(timeout)
            t0 = time()
            try:
                estimator.fit(X_train, y_train)
            except TimeoutException:
                print(
                    f'    Warning: {estimator_name} timed out after {timeout}s.'
                )
                score_total += 0
                time_total += timeout
                continue

            t1 = time()
            #signal.alarm(0)

            score = f1_score(
                y_test, estimator.predict(X_test), average='micro'
            )
            score_total += score
            time_total += t1 - t0

        scores[dataset_name][estimator_name] = score_total / n_folds
        times[dataset_name][estimator_name] = time_total / n_folds
        print('done with dataset', dataset_name) # TODO remove


def main():
    scores = {}
    times = {}

    # fill the scores and times with dummy elements to avoid reallocations
    # during multiprocessing
    for dataset_name in datasets:
        scores[dataset_name] = {}
        times[dataset_name] = {}
        for estimator_name in estimators:
            scores[dataset_name][estimator_name] = 0
            times[dataset_name][estimator_name] = 0

    # evaluate estimators in parallel
    n_workers = min(os.process_cpu_count(), len(datasets) - 1)
    if n_workers <= 0:
        print('Nothing to do')
        sys.exit(0)

    print(f'{len(datasets)} datasets to process, '
          f'splitting work in {n_workers} workers.')

    pool = Pool(n_workers)
    results = []
    for dataset_name in datasets:
        res = pool.apply_async(worker, (scores, times, dataset_name))
        results.append(res)

    for res in results:
        res.get()
    pool.close()
    pool.join()

    print('Done.')
    save_data(scores, times) # TODO remove

if __name__ == '__main__':
    main()
