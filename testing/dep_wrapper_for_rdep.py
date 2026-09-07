import numpy as np
from typing import cast, Protocol
from sklearn_morpho import DEP
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class FitMixin(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> ClassifierMixin: ...
    def decision_function(self, X: np.ndarray) -> np.ndarray: ...


class EnsembleTransform(TransformerMixin, BaseEstimator):
    def __init__(self, estimators: list[FitMixin]) -> None:
        self.estimators = estimators

    def fit(self, X, y):
        for e in self.estimators:
            e.fit(X, y)
        return self

    def transform(self, X):
        return np.vstack(
            [estimator.decision_function(X) for estimator in self.estimators]
        ).T


def colorize(y_real: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.where(y_real == y_pred, y_real, np.repeat(['#aaa'], y_real.shape))


random_state = np.random.RandomState(11)
X, y = make_moons(n_samples=5000, noise=0.2, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)
X_train, X_test = cast(np.ndarray, X_train), cast(np.ndarray, X_test)
y_train, y_test = cast(np.ndarray, y_train), cast(np.ndarray, y_test)

svcs: list[FitMixin] = [SVC(kernel='rbf'), SVC(kernel='linear')]
ensemble = EnsembleTransform(svcs)
dep = DEP(random_state=random_state)
rdep = make_pipeline(ensemble, dep)

rdep.fit(X_train, y_train)
y_pred = rdep.predict(X_test)
accuracy_str = (
    f'Accuracy: {round((y_test == y_pred).sum() / len(y_test) * 100)}%'
)


colors = np.array('red blue'.split())
disp = DecisionBoundaryDisplay.from_estimator(
    rdep,
    X_test,
    response_method='decision_function',
    grid_resolution=200,
    plot_method='contour',
    levels=[0],
    colors='black',
)
ax = disp.ax_
ax.scatter(*X_test.T, color=colorize(colors[y_test], colors[y_pred]))
ax.title.set_text(accuracy_str)
plt.show()
