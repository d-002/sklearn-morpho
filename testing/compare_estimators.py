"""
Create and train different estimators and display their respective scores
averaged from multiple datasets.
Estimators selection inspired by arxiv/2011.06512
"""

import json
import signal
import numpy as np
from time import time
from scipy.sparse._csr import csr_matrix

from sklearn.metrics import f1_score
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold

# perceptrons
from sklearn.svm import LinearSVC, SVC
from sklearn.impute import SimpleImputer
from sklearn_morpho import LDEP
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

FILE = 'comparison_data.json'
n_splits = 5
timeout = 60

# set up estimators and datasets
random_state = np.random.RandomState()
print(f'Random state: {random_state}')

estimators = {
    'l_DEP': OneVsRestClassifier(LDEP(random_state=random_state)),
    'Linear SVC': LinearSVC(random_state=random_state),
    'RBF SVC': SVC(kernel='rbf', random_state=random_state),
    'MLP': MLPClassifier(max_iter=1000, random_state=random_state),
    'Poly SVC': SVC(kernel='poly', random_state=random_state),
}

def get_clean_openml(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    kwargs.setdefault('as_frame', False)
    kwargs.setdefault('version', 1)
    X, y = fetch_openml(name, return_X_y=True, **kwargs)

    # make sure X and y are not sparse
    if type(X) == csr_matrix:
        X = X.toarray()
    if type(y) == csr_matrix:
        y = y.toarray()

    # make sure X contains only numbers, not yes/no like in australian
    oe = OrdinalEncoder()
    X = oe.fit_transform(X)

    return X, y

datasets_names = [
    'acute-inflammations', 'australian', 'banana', 'banknote-authentication',
    'blood-transfusion-service-center', 'breast-cancer', 'chess', 'colic',
    'credit-approval', 'credit-g', 'cylinder-bands', 'diabetes',
    'eeg-eye-state', 'haberman', 'hill-valley', 'ilpd', 'ionosphere',
    'mofn-3-7-10', 'monks-problems-2', 'mushroom', 'phoneme', 'PhishingWebsites',
    'sick', 'sonar', 'spambase', 'steel-plates-fault', 'thoracic-surgery',
    'tic-tac-toe', 'titanic',
]
temp_blacklist = ['chess', 'eeg-eye-state', 'PhishingWebsites', 'sick',
                  'spambase']
for name in temp_blacklist:
    datasets_names.remove(name)

# datasets not included:
# - chess (slow for LDEP)
# - eeg-eye-state (slow for LDEP)
# - internet-advertisements (internally referring to a nonexistent dataset)
# - PhishingWebsites (slow for LDEP)
# - sick (slow for LDEP)
# - spambase (slow for LDEP)
datasets_options = {
    'australian': { 'version': 4 },
    'Breast_Cancer_Wisconsin': { 'as_frame': True },
    'cylinder-bands': { 'version': 6 },
    'titanic': { 'as_frame': True },
}

# evaluate estimators
scores = {}
times = {}

class TimeoutException(Exception):pass
def timeout_handler(signum, frame):
    raise TimeoutException('Timed out')
signal.signal(signal.SIGALRM, timeout_handler)

def save_data():
    with open(FILE, 'w') as f:
        json.dump({ 'n_splits': n_splits, 'scores': scores, 'times': times }, f,
                  indent=2)

skf = StratifiedKFold(n_splits=n_splits)
for dataset_name in datasets_names:
    print(f"Training with dataset '{dataset_name}'...")
    scores[dataset_name] = {}
    times[dataset_name] = {}

    # trying to factorize the code but some datasets must be loaded differently
    match dataset_name:
        case 'breast-cancer':
            X, y = load_breast_cancer(return_X_y=True)
        case _:
            X, y = get_clean_openml(dataset_name,
                                    **datasets_options.get(dataset_name, {}))

    for estimator_name, estimator in estimators.items():
        print(f'  - Estimator {estimator_name}...')
        scores[dataset_name][estimator_name] = []
        times[dataset_name][estimator_name] = []

        estimator = make_pipeline(
            SimpleImputer(strategy='mean'), # remove NaNs
            estimator
        )

        for X_fold, y_fold in skf.split(X, y):
            X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=.3, random_state=random_state)

            signal.alarm(timeout)
            t0 = time()
            try:
                estimator.fit(X_train, y_train)
            except TimeoutException:
                print(f'    Warning: {estimator_name} timed out after '
                      f'{timeout}s.')
                scores[dataset_name][estimator_name].append(0)
                times[dataset_name][estimator_name].append(timeout)
                continue

            t1 = time()
            signal.alarm(0)

            score = f1_score(y_train, estimator.predict(X_train),
                             average='micro')
            scores[dataset_name][estimator_name].append(score)
            times[dataset_name][estimator_name].append(t1 - t0)

    save_data()

print('Done.')
