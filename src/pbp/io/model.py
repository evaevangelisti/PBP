"""Save and load fitted classifiers."""

import pickle
from pathlib import Path

from ..training import Classifier


def save_model(
    path: Path,
    model: Classifier,
) -> None:
    """Serialize a classifier.

    Args:
        path: Destination path for the serialized classifier.
        model: Classifier to save, including its fitted state.

    Raises:
        OSError: If the destination file cannot be opened or written.
        pickle.PicklingError: If the classifier cannot be serialized.
    """
    with path.open("wb") as stream:
        pickle.dump(
            model,
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
