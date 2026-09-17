from os import makedirs
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn_morpho import DEP, LDEP, RDEP, MorphoPerceptron

OUTPUT_DIR = 'dgmm_27_figures'
makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(fig: Figure, filename: str) -> None:
    fig.savefig(
        join(OUTPUT_DIR, filename),
        transparent=True,
        bbox_inches='tight',
        pad_inches=0,
        dpi=300,
    )


print('Initalizing and training perceptrons')

random_state = np.random.RandomState(42)
X, y = make_moons(n_samples=500, noise=0.05, random_state=random_state)
y = np.array(['blue', 'red'])[y]

morpho = MorphoPerceptron(random_state=random_state)
dep = DEP(random_state=random_state)
rdep = RDEP(random_state=random_state)
ldep = LDEP(random_state=random_state)
morpho.fit(X, y)
dep.fit(X, y)
rdep.fit(X, y)
ldep.fit(X, y)

figname = 'morpho_vs_dep_two_moons.svg'
print('Rendering figure:', figname)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_axis_off()
ax.scatter(*X.T, color=y, alpha=0.2)
ax.scatter(*X.T, color=y)
DecisionBoundaryDisplay.from_estimator(
    morpho,
    X,
    plot_method='contour',
    response_method='predict',
    grid_resolution=1000,
    colors='gray',
    ax=ax,
    linewidths=2,
)
DecisionBoundaryDisplay.from_estimator(
    dep,
    X,
    plot_method='contour',
    response_method='predict',
    grid_resolution=1000,
    colors='black',
    ax=ax,
    linewidths=2,
)
savefig(fig, figname)

figname = 'rdep_vs_ldep_two_moons.svg'
print('Rendering figure:', figname)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_axis_off()
ax.scatter(*X.T, color=y, alpha=0.2)
ax.scatter(*X.T, color=y)
DecisionBoundaryDisplay.from_estimator(
    rdep,
    X,
    plot_method='contour',
    response_method='predict',
    grid_resolution=1000,
    colors='gray',
    ax=ax,
    linewidths=2,
)
DecisionBoundaryDisplay.from_estimator(
    ldep,
    X,
    plot_method='contour',
    response_method='predict',
    grid_resolution=1000,
    colors='black',
    ax=ax,
    linewidths=2,
)
savefig(fig, figname)
