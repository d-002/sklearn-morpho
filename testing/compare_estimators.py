"""
Create and train different estimators and display their respective scores
averaged from multiple datasets.

Estimators selection inspired by arxiv/2011.06512
"""

import json
import warnings
from time import time
from types import FrameType
from concurrent.futures import ProcessPoolExecutor, TimeoutError

import numpy as np
from scipy.sparse._csr import csr_matrix
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC, LinearSVC

# perceptrons
from sklearn_morpho import DEP, LDEP, RDEP, MorphoPerceptron
from sklearn_morpho.training import SOLVER_DCCP
from sklearn_morpho.utils import Kind

FILE = 'comparison.json'
n_folds = 5
timeout = 60

# set up estimators and datasets
random_state = np.random.RandomState()
print(f'Random state: {random_state}')

estimators = {
    'l-DEP': OneVsRestClassifier(LDEP(random_state=random_state)),
    'DCCP l-DEP': OneVsRestClassifier(
        LDEP(solver=SOLVER_DCCP, random_state=random_state)
    ),
    'r-DEP': OneVsRestClassifier(RDEP(random_state=random_state)),
    'DCCP r-DEP': OneVsRestClassifier(
        RDEP(solver=SOLVER_DCCP, random_state=random_state)
    ),
    'DEP': OneVsRestClassifier(DEP(random_state=random_state)),
    'DCCP DEP': OneVsRestClassifier(
        DEP(solver=SOLVER_DCCP, random_state=random_state)
    ),
    'Morpho_max': OneVsRestClassifier(
        MorphoPerceptron(kind=Kind.MAX, random_state=random_state)
    ),
    'Morpho_max_DCCP': OneVsRestClassifier(
        MorphoPerceptron(
            kind=Kind.MAX, solver=SOLVER_DCCP, random_state=random_state
        )
    ),
    'Morpho_min': OneVsRestClassifier(
        MorphoPerceptron(kind=Kind.MIN, random_state=random_state)
    ),
    'Morpho_min_DCCP': OneVsRestClassifier(
        MorphoPerceptron(
            kind=Kind.MIN, solver=SOLVER_DCCP, random_state=random_state
        )
    ),
    'Linear SVC': LinearSVC(random_state=random_state),
    'RBF SVC': SVC(kernel='rbf', random_state=random_state),
    'MLP': MLPClassifier(random_state=random_state),
    'Poly SVC': SVC(kernel='poly', random_state=random_state),
}


def get_clean_openml(
    name: str, **kwargs: str | bool | int
) -> tuple[np.ndarray, np.ndarray]:
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


datasets_names = [
    'acute-inflammations',
    'australian',
    'banana',
    'banknote-authentication',
    'blood-transfusion-service-center',
    'breast-cancer',
    # 'chess',
    'colic',
    'credit-approval',
    'credit-g',
    'cylinder-bands',
    'diabetes',
    # 'eeg-eye-state',
    'haberman',
    'hill-valley',
    'ilpd',
    # 'internet-advertisements',
    'ionosphere',
    'mofn-3-7-10',
    'monks-problems-2',
    'mushroom',
    'phoneme',
    # 'PhishingWebsites',
    # 'sick',
    'sonar',
    # 'spambase',
    'steel-plates-fault',
    'thoracic-surgery',
    'tic-tac-toe',
    'titanic',
]

datasets_options: dict[str, dict[str, bool | int]] = {
    'australian': {'version': 4},
    'Breast_Cancer_Wisconsin': {'as_frame': True},
    'cylinder-bands': {'version': 6},
    'titanic': {'as_frame': True},
}

# evaluate estimators
scores: dict[str, dict[str, list[float]]] = {}
times: dict[str, dict[str, list[float]]] = {}


def train_pair(estimator, X, y):
    scores = []
    times = []

    for i_train, i_test in skf.split(X, y):
        X_train, X_test = X[i_train], X[i_test]
        y_train, y_test = y[i_train], y[i_test]

        t0 = time()
        estimator.fit(X_train, y_train)
        t1 = time()

        score = f1_score(y_train, estimator.predict(X_train), average='micro')
        scores.append(score)
        times.append(t1 - t0)

    return scores, times


def save_data() -> None:
    with open(FILE, 'w') as f:
        json.dump(
            {'n_folds': n_folds, 'scores': scores, 'times': times},
            f,
            indent=2,
        )


skf = StratifiedKFold(n_splits=n_folds)
for dataset_name in datasets_names:
    print(f'Training with dataset "{dataset_name}"...')
    scores[dataset_name] = {}
    times[dataset_name] = {}

    # trying to factorize the code but some datasets must be loaded differently
    match dataset_name:
        case 'breast-cancer':
            X, y = load_breast_cancer(return_X_y=True)
        case _:
            X, y = get_clean_openml(
                dataset_name, **datasets_options.get(dataset_name, {})
            )

    for estimator_name, estimator in estimators.items():
        print(f'  - Estimator {estimator_name}...')

        estimator = make_pipeline(
            SimpleImputer(strategy='mean'),  # remove NaNs
            estimator,
        )

        # run in a different process to avoid signals being ignored in the C/C++
        # solver layers
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(train_pair, estimator, X, y)

            t0 = time()  # for timeout error message calculation
            try:
                scores_pair, times_pair = future.result(
                    timeout=timeout * n_folds
                )
                scores[dataset_name][estimator_name] = scores_pair
                times[dataset_name][estimator_name] = times_pair
            except TimeoutError:
                warnings.warn(
                    f'{estimator_name} timed out after {time() - t0}s.'
                )

    save_data()

print('Done.')
