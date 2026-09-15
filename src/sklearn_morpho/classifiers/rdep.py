from __future__ import annotations

from typing import Literal, Protocol, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import Tags, check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from ..inversion import CentroidInversion, InversionHeuristic
from ..stopping import (
    CostStoppingMethod,
    EarlyStoppingMethod,
    EpochStoppingMethod,
    StoppingMethod,
    TrainStopStoppingMethod,
)
from ..training.dccp_dep import DEPDccpTrainer
from ..weighting import NoneSampleWeighting, SampleWeighting
from .dep import DEP


class FitMixin(Protocol):
    """
    Simple class used as an interface for duck typing to be able to cleanly type
    a list of estimators.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> ClassifierMixin: ...
    def decision_function(self, X: np.ndarray) -> np.ndarray: ...


class EnsembleTransform(TransformerMixin, BaseEstimator):
    """
    Transformer for the chosen r-DEP preprocessing.
    See help(RDEP) for more.
    """

    def __init__(self, estimators: list[FitMixin]) -> None:
        self.estimators = estimators

    def fit(self, X: np.ndarray, y: np.ndarray) -> EnsembleTransform:
        for e in self.estimators:
            e.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.vstack(
            [estimator.decision_function(X) for estimator in self.estimators]
        ).T


class RDEP(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn estimator wrapper around a r-DEP (Reduced Dilation-Erosion
    morphological Perceptron) for binary data classification.

    The rDEP's activation function is defined as:

    \\[ y = f(\\lambda \\tau_(\\rho(x)) + (1 - \\lambda) \\tau'_(\\rho(x))) \\]

    Where $\\tau$ refers to the activation of a (max, +) morphological
    perceptron, $\\tau'$ to a (min, +) one, and $\\rho$ is a surjective mapping.

    $\\lambda$ is a real number between 0 and 1 to guarantee correct convexity,
    but in practice a smaller interval can be enforced to avoid imprecisions.

    Here $\\rho$ is proposed to be computed using the activation function of
    multiple estimators: it would act as a transformation from the input space
    to the latent $n$-dimensional space where the position of an input data in
    one dimension corresponds to its activation by one of the classifiers.
    The motivation behind this is that a DEP behaves well only if the input data
    follows a specific ordering, which this transformation helps with.

    In practice, the r-DEP is implemented as a sklearn pipeline with:
    1. A transformation of the data from the input space to the latent space
       described above
    2. A classification using a DEP
    """

    def __init__(
        self,
        preprocessing_estimators: list[FitMixin] = [
            SVC(kernel='rbf'),
            SVC(kernel='linear'),
        ],
        lambda_bounds: tuple[float, float] = (1e-3, 1 - 1e-3),
        margin: float = 0.0,
        penalty: float = 0.0,
        validation_ratio: float = 0.3,
        weighting_method: SampleWeighting | None = None,
        stopping_methods: list[StoppingMethod] | None = None,
        inversion_method: InversionHeuristic | None = None,
        solver: str | None = None,
        verbose: Literal[0, 1, 2] = 0,
        random_state: np.random.RandomState | None = None,
    ) -> None:
        """
        Initialize the classifier, see help(DEP) for more.

        - param `lambda_bounds`:
          A pair of min and max values for lambda, to avoid solvers (especially
          dccp) from failing to optimize.
          To keep the constraints at the right convexity, the bounds must be
          inside [0, 1].
        - param `margin`:
          Enforce a margin between the decision boundary and the data.
          May help with linearly separable datasets, but generally lower is more
          accurate.
        - param `penalty`:
          A penalty to add to the weights squared and avoid them exploding.
          Must be a small positive number like 1e-6, or zero to disable penalty
          calculation altogether.
        - param `validation_radio`:
          How much of the training set to dedicate to use as validation during
          fitting.
          Must be between 0 and 1 (inclusive, exclusive), if set to exactly 0
          then incompatible stopping methods cannot be used (e.g. early
          stopping).
          Ignored when using the dccp library solver.
        - param `weighting_method`:
          The weighting method to use: apply weights to the cost contribution of
          each data point to help avoid outliers.
          If left to None, will use NoneWeightingMethod().
        - param `stopping_methods`:
          A list of stopping methods, must not be empty.
          At each epoch, these methods will be sequentially asked whether the
          training should stop.
          In this case, epoch ends by rolling back to the epoch with the best
          validation cost.

          If left to None, will use:
          ```
          [
              CostStoppingMethod(),
              EarlyStoppingMethod(),
              EpochStoppingMethod(),
              TrainStopStoppingMethod(),
          ]
          ```

          Ignored when using the dccp library solver.
        - param `inversion_method`:
          The heuristic to use to know whether to invert the target classes, as
          the dataset's orientation might not always be favorable.
          If left to None, will use a CentroidInversion optimizing for
          (1, 1, ..., 1).
        - param `solver`:
          The solver to use in cvxpy optimization.
          If set to "dccp", will use the solver from the dccp library instead of
          the customized DCA.
        - param `verbose`:
          Whether to log extra information.

          - 0: no logging
          - 1: basic logging / timing
          - 2: cvxpy solve() set to verbose mode.
        - param `random_state`:
          A RandomState object or None to allow for seeded randomness.
        """

        self.preprocessing_estimators = preprocessing_estimators
        # DEP parameters
        self.lambda_bounds = lambda_bounds
        self.margin = margin
        self.penalty = penalty
        self.validation_ratio = validation_ratio
        self.weighting_method = weighting_method
        self.stopping_methods = stopping_methods
        self.inversion_method = inversion_method
        self.solver = solver
        self.verbose: Literal[0, 1, 2] = verbose
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> RDEP:
        """
        Fit the classifier, create attributes:
        - self.ensemble_: preprocessing transformation
        - self.dep_: inner DEP
        - self.pipeline_: complete pipeline forming the r-DEP logic
        """

        # Do not check data integrity, this will already be done in the child
        # estimators

        # initialize estimators
        self.ensemble_ = EnsembleTransform(self.preprocessing_estimators)
        self.dep_ = DEP(
            self.lambda_bounds,
            self.margin,
            self.penalty,
            self.validation_ratio,
            self.weighting_method,
            self.stopping_methods,
            self.inversion_method,
            self.solver,
            self.verbose,
            self.random_state,
        )
        self.pipeline_ = make_pipeline(
            self.ensemble_,
            self.dep_,
        )

        # fit estimators
        self.pipeline_.fit(X, y)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        res: np.ndarray = self.pipeline_.decision_function(X)
        return res

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        res: np.ndarray = self.pipeline_.predict(X)
        return res
