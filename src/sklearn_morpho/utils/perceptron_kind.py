import enum


class Kind(str, enum.Enum):
    """
    The kind of morphological perceptron, either min or max.
    Used for the simple_perceptron classifier.
    """

    MIN = 'min'
    MAX = 'max'
