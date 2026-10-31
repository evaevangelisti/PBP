"""Train a configured classifier."""

from collections.abc import Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression

type Classifier = LogisticRegression


def train_model[Model: Classifier](
    model: Model,
    features: pd.DataFrame,
    targets: Sequence[int],
) -> Model:
    """Fit a configured classifier on the supplied training examples.

    Args:
        model: Classifier whose configuration has already been selected.
        features: Numerical feature table with one row per battle.
        targets: Binary outcomes aligned with the feature rows.

    Returns:
        The supplied classifier, fitted in place.

    Raises:
        ValueError: If the classifier rejects the parameters or training data.
    """
    model.fit(features, targets)

    return model
