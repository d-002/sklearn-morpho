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

# display summary table in console
datasets_names = list(scores.keys())
estimators_names = list(scores[datasets_names[0]].keys())

for dataset_name in datasets_names:
    for estimator_name in estimators_names:
        for data_source in (scores, times):
            arr = data_source[dataset_name][estimator_name]
            data_source[dataset_name][estimator_name] = np.array(arr)

print()
params = (
    ('F1 score', scores, np.argmax, np.argmin),
    ('Time (s)', times, np.argmin, np.argmax),
)
for name, data_source, best_func, worst_func in params:
    headers = ''
    for estimator_name in estimators_names:
        headers += f' | {estimator_name:<16}'
    header = f'{name:>35}' + headers
    print(header)
    print('=' * len(header))
    for dataset_name in datasets_names:
        line = ''

        dataset_res = []  # list of (avg, std)
        for estimator_name in estimators_names:
            res = data_source[dataset_name][estimator_name]

            avg = np.average(res)
            std = res.std()
            dataset_res.append((avg, std, len(res)))

        best = best_func([avg for avg, _, _ in dataset_res])
        worst = worst_func([avg for avg, _, _ in dataset_res])
        for i, (avg, std, length) in enumerate(dataset_res):
            fail = '' if length == 5 else f' ({5 - length}F)'
            chunk = f'{avg:.2f}±{std:.2f}{fail}'
            chunk = f'{chunk:<16}'
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
            for score in scores[dataset_name][estimator_name]
        ]
    )
    estimators_times[estimator_name] = np.array(
        [
            time
            for dataset_name in datasets_names
            for time in times[dataset_name][estimator_name]
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
