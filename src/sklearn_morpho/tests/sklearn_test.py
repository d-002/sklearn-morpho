from sklearn.utils.estimator_checks import check_estimator

from sklearn_morpho.classifiers.ldep import LDEP
from sklearn_morpho.stopping import (
        CostStoppingMethod,
        IterStoppingMethod,
        TrainStopStoppingMethod,
)


def test_check_estimator():
    ldep = LDEP(validation_ratio=0,
                stopping_methods=[
                    CostStoppingMethod(1e-6),
                    IterStoppingMethod(20),
                    TrainStopStoppingMethod(),
                ])

    assert check_estimator(ldep)
