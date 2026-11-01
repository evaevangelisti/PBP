"""Predict battle outcomes."""

import pandas as pd

from .training import Classifier


def predict(
    model: Classifier,
    features: pd.DataFrame,
) -> list[int]:
    """Predict outcomes.

    Args:
        model: Trained classifier.
        features: Prediction features.

    Returns:
        Binary predictions in row order.

    Raises:
        sklearn.exceptions.NotFittedError: If the classifier has not been fitted.
        ValueError: If input features are invalid.
    """
    return [int(value) for value in model.predict(features)]
