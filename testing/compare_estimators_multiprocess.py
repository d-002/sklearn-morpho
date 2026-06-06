"""
Create and train different estimators and display their respective scores
averaged from multiple datasets.
Train perceptrons in parallel using multiple processes, so timing information
may be inaccurate while scoring should stay the same and will take less time.
Estimators selection inspired by arxiv/2011.06512
"""

import os
import sys
import json
import time
import signal
import warnings
import numpy as np

from time import time, sleep
from threading import Thread
from multiprocessing import Manager
from joblib import Parallel, delayed
from scipy.sparse._csr import csr_matrix

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

## Timeouts handling


class TimeoutException(Exception):
    pass


def timeout_handler(sugnum, frame):
    raise TimeoutException()


## Configuration

FILE = 'comparison.json'
# ignore warnings for cleaner logs
ignore_warnings = True
n_folds = 5
max_jobs = 15
random_state = np.random.RandomState()
LINE_SIZE = 100
timeout = 60

if ignore_warnings:
    warnings.filterwarnings('ignore')

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
    #'DCCP l_DEP': OneVsRestClassifier(
    #    LDEP(use_dccp_library=True, random_state=random_state)
    # ),
    'r-DEP': OneVsRestClassifier(RDEP(random_state=random_state)),
    #'DCCP r_DEP': OneVsRestClassifier(
    #    RDEP(use_dccp_library=True, random_state=random_state)
    # ),
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

## Load datasets
print('\nLoading datasets...')

datasets = {}
for i, dataset_name in enumerate(datasets_names):
    print(
        f'\033[32m=>\033[m Loading {dataset_name:<35} '
        f'({i:02}/{len(datasets_names):02})',
        end='\r',
    )
    match dataset_name:
        case 'breast-cancer':
            datasets[dataset_name] = load_breast_cancer(return_X_y=True)
        case _:
            datasets[dataset_name] = get_clean_openml(
                dataset_name, **datasets_options.get(dataset_name, {})
            )
print(' ' * LINE_SIZE)

## Main logic

# centralized data
manager = Manager()
scores = manager.dict({d: {e: [] for e in estimators} for d in datasets_names})
times = manager.dict({d: {e: [] for e in estimators} for d in datasets_names})

# the elements are a list of:
# - start timestamp, or 0 if not started
# - number from 0 to 1 for progress, or a timestamp if ended
progress_states = manager.dict(
    {d: {e: [0.0, 0.0] for e in estimators} for d in datasets_names}
)
total_jobs = len(datasets) * len(estimators)


def worker(dataset_name: str, estimator_name: str) -> None:
    if ignore_warnings:
        warnings.filterwarnings('ignore')

    score_arr = []
    time_arr = []
    start = time()

    # need to reassign because of shared data getters/setters
    temp = progress_states[dataset_name]
    temp[estimator_name][0] = start
    progress_states[dataset_name] = temp

    X, y = datasets[dataset_name]
    estimator = make_pipeline(
        SimpleImputer(strategy='mean'),  # remove NaNs
        estimators[estimator_name],
    )

    for i, (i_train, i_test) in enumerate(skf.split(X, y)):
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            X_train, X_test = X[i_train], X[i_test]
            y_train, y_test = y[i_train], y[i_test]

            t0 = time()
            estimator.fit(X_train, y_train)
            t1 = time()

            score = f1_score(y_test, estimator.predict(X_test), average='micro')
            score_arr.append(score)
            time_arr.append(t1 - t0)

            temp = progress_states[dataset_name]
            temp[estimator_name][1] = (i + 1) / n_folds
            progress_states[dataset_name] = temp
        except TimeoutException:
            break

    for data_source, value in ((scores, score_arr), (times, time_arr), (progress_states, [start, time()])):
        temp = data_source[dataset_name]
        temp[estimator_name] = value
        data_source[dataset_name] = temp
    signal.alarm(0)


def save_data():
    with open(FILE, 'w') as f:
        data = {
            'n_folds': n_folds,
            'scores': dict(scores),
            'times': dict(times),
        }
        json.dump(data, f, indent=2)


def progress_bar():
    print('=' * LINE_SIZE)
    print('\n')
    prev_running = 0

    def format_time(t):
        hm, s = divmod(int(t), 60)
        h, m = divmod(hm, 60)
        return f'\033[34m{h:02}:{m:02}:{s:02}\033[m'

    def eta_str(progress, time_spent):
        if not progress:
            return f'--:--:--'

        eta = time_spent / progress * (1 - progress)
        return format_time(eta)

    def bar(name, time_spent, progress):
        print(
            f'[{"#" * round(progress * 30):-<30}] '
            f'(\033[33m{int(progress * 100):>3}%\033[m) '
            f'{format_time(time_spent)} {name:<50}|'
        )

    while True:
        running = []
        total_time = 0
        total_progress = 0
        states = []
        for dataset_name, elt in dict(progress_states).items():
            for estimator_name, (start, progress) in elt.items():
                if progress > 1:
                    states.append(2)
                    total_time += progress - start
                elif start != 0:
                    states.append(1)
                    total_time += time() - start
                else:
                    states.append(0)
                if start != 0:
                    if progress < 1:
                        running.append(
                            (dataset_name, estimator_name, start, progress)
                        )
                        total_progress += progress
                    else:
                        total_progress += 1

        total_done, total_running, total_queued = (states.count(k) for k in [2, 1, 0])
        total_progress /= total_jobs
        running.sort(key=lambda x: x[3])
        now_running = len(running)

        for i in range(prev_running + 1, -1, -1):
            if i < now_running:
                print(end='\033[F')
            else:
                print(' ' * LINE_SIZE, end='\033[F')

        for dataset_name, estimator_name, start, progress in running:
            bar(f'{dataset_name} - {estimator_name}', time() - start, progress)

        print('=' * LINE_SIZE)
        print(
            f'TOTAL [{"#" * round(total_progress * 24):-<24}] '
            f'(\033[33m{int(total_progress * 100):>3}%\033[m) '
            f'ETA: {eta_str(total_progress, total_time)} - '
            f'{total_done} done, {total_running} running, {total_queued} queued'
        )

        # a job was just done, update the results file
        if now_running != prev_running:
            save_data()

        if total_done == total_jobs:
            break

        prev_running = now_running
        sleep(0.2)


## train in parallel
print('Preparing to train')


n_jobs = min(os.process_cpu_count(), total_jobs, max_jobs)
if n_jobs <= 0:
    print('Nothing to do')
    sys.exit(0)

print(f'{len(datasets)} datasets to process, splitting work in {n_jobs} jobs.')
print()

progress_bar_thread = Thread(target=progress_bar)
progress_bar_thread.start()

Parallel(n_jobs=n_jobs)(
    delayed(worker)(dataset_name, estimator_name)
    for dataset_name in datasets
    for estimator_name in estimators
)

# force end all datasets
for d in datasets:
    for e in estimators:
        progress_states[e][d][0] = time()
        progress_states[e][d][1] = time()
progress_bar_thread.join()

scores, times = dict(scores), dict(times)

print('Done.')
save_data()
