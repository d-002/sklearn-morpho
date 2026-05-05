"""
Display training results gotten from a compare_estimators.py run.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

FILE = 'comparison_data.json'

with open(FILE, 'r') as f:
    data = json.load(f)
scores = data['scores']
times = data['times']

n_splits = data['n_splits']

# display summary table in console
datasets_names = list(scores.keys())
estimators_names = list(scores[datasets_names[0]].keys())

print()
params = (('F1 score', scores, np.argmax, np.argmin),
          ('Time (s)', times, np.argmin, np.argmax))
for name, data_source, best_func, worst_func in params:
    headers = ''
    for estimator_name in estimators_names:
        headers += f' | {estimator_name:<12}'
    header = f'{name:>35}' + headers
    print(header)
    print('=' * len(header))
    for dataset_name in datasets_names:
        line = ''

        dataset_res = [] # list of (avg, std)
        for estimator_name in estimators_names:
            res = np.array(data_source[dataset_name][estimator_name])

            avg = np.average(res)
            std = res.std()
            dataset_res.append((avg, std))

        best = best_func([avg for avg, std in dataset_res])
        worst = worst_func([avg for avg, std in dataset_res])
        for i, (avg, std) in enumerate(dataset_res):
            chunk = f'{avg:.2f}±{std:.2f}'
            chunk = f'{chunk:<12}'
            if i == best:
                chunk = f'\033[32m{chunk}\033[m'
            if i == worst:
                chunk = f'\033[31m{chunk}\033[m'
            line += f' | {chunk}'

        print(f'{dataset_name:>35}' + line)
    print()

# merge and display results averaged over all datasets
estimators_scores = {}
estimators_times = {}
for estimator_name in estimators_names:
    estimators_scores[estimator_name] = np.empty(len(datasets_names))
    estimators_times[estimator_name] = np.empty(len(datasets_names))

    for i, dataset_name in enumerate(datasets_names):
        score = np.array(scores[dataset_name][estimator_name])
        time = np.array(times[dataset_name][estimator_name])

        estimators_scores[estimator_name][i] = np.average(score)
        estimators_times[estimator_name][i] = np.average(time)

fig, axs = plt.subplots(ncols=2, nrows=1)
# log scale for times
axs[1].set_yscale('log')

for data_source, name, ax in zip((estimators_scores, estimators_times),
                                 ('F1 score', 'Training time (s)'), axs):
    ax.set_title(name)
    ax.boxplot(data_source.values(), patch_artist=True,
               tick_labels=data_source.keys())
    ax.tick_params('x', rotation=90)

plt.show()
