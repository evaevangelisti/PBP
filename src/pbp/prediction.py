"""Predict outcomes with a fitted classifier."""

import pandas as pd

from .training import Classifier


def predict(
    model: Classifier,
    features: pd.DataFrame,
) -> list[int]:
    """Predict binary outcomes with a fitted classifier.

    Args:
        model: Classifier fitted on the same feature representation.
        features: Numerical feature table with one row per battle.

    Returns:
        Binary predictions in feature row order.

    Raises:
        sklearn.exceptions.NotFittedError: If the classifier has not been fitted.
        ValueError: If the classifier rejects the input features.
    """
    return [int(value) for value in model.predict(features)]
