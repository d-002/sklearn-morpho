"""
Display training results gotten from a compare_estimators.py run.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

FILE = 'comparison.json'

with open(FILE, 'r') as f:
    data = json.load(f)
scores = data['scores']
times = data['times']

n_folds = data['n_folds']

datasets_names = list(scores.keys())
# because of timeouts, check all the datasets to get the list of estimators
estimators_set: set[str] = set()
for results in scores.values():
    estimators_set = estimators_set.union(set(results.keys()))
# sort heuristic to make it look nicer: use suffixes
estimators_names = sorted(
    estimators_set, key=lambda name: name[::-1], reverse=True
)

# display summary table in console

print()
params = (
    ('F1 score', scores, np.argmax, np.argmin),
    ('Time (s)', times, np.argmin, np.argmax),
)
for name, data_source, best_func, worst_func in params:
    headers = ''
    for estimator_name in estimators_names:
        headers += f' | {estimator_name:<15}'
    header = f'{name:>35}' + headers
    print(header)
    print('=' * len(header))
    for dataset_name in datasets_names:
        line = ''

        dataset_res: list[tuple[float, float] | None] = []  # list of (avg, std)
        for estimator_name in estimators_names:
            res = data_source[dataset_name].get(estimator_name)

            if res is None:
                dataset_res.append(None)
            else:
                arr = np.array(res)
                avg = np.average(arr)
                dataset_res.append((np.average(arr), arr.std()))

        best = best_func([res[0] for res in dataset_res if res is not None])
        worst = worst_func([res[0] for res in dataset_res if res is not None])
        for i, res in enumerate(dataset_res):
            if res is None:
                fail = f'TIMEOUT'
                chunk = ''
            else:
                fail = ''
                avg, std = res
                chunk = f'{avg:.2f}±{std:.2f}{fail}'
            chunk = f'{chunk:<15}'
            if i == best:
                chunk = f'\033[32m{chunk}\033[m'
            if i == worst:
                chunk = f'\033[31m{chunk}\033[m'
            line += f' | {chunk}'

        print(f'{dataset_name:>35}' + line)
    print()

# merge and display results averaged over all datasets where there are results
estimators_scores = {}
estimators_times = {}
for estimator_name in estimators_names:
    estimators_scores[estimator_name] = np.array(
        [
            score
            for dataset_name in datasets_names
            for score in scores[dataset_name].get(estimator_name, [])
        ]
    )
    estimators_times[estimator_name] = np.array(
        [
            time
            for dataset_name in datasets_names
            for time in times[dataset_name].get(estimator_name, [])
        ]
    )

fig, axs = plt.subplots(ncols=2, nrows=1)
# log scale for times
axs[1].set_yscale('log')

for data_source, name, ax in zip(
    (estimators_scores, estimators_times),
    ('F1 score', 'Training time (s)'),
    axs,
):
    ax.set_title(name)
    ax.boxplot(
        data_source.values(), patch_artist=True, tick_labels=data_source.keys()
    )
    ax.tick_params('x', rotation=90)

plt.show()
