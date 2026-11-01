"""Train classifiers."""

from collections.abc import Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression

type Classifier = LogisticRegression


def train_model[Model: Classifier](
    model: Model,
    features: pd.DataFrame,
    targets: Sequence[int],
) -> Model:
    """Train a classifier.

    Args:
        model: Classifier to train.
        features: Training features.
        targets: Training targets in row order.

    Returns:
        Trained classifier.

    Raises:
        ValueError: If data or parameters are invalid.
    """
    model.fit(features, targets)

    return model
