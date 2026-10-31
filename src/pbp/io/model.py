"""Save and load fitted classifiers."""

import pickle
from pathlib import Path

from ..training import Classifier


def save_model(
    path: Path,
    model: Classifier,
) -> None:
    """Serialize a classifier to a new pickle file.

    Args:
        path: Destination path for the serialized classifier.
        model: Classifier to save, including its fitted state.

    Raises:
        FileExistsError: If the destination file already exists.
        OSError: If the destination file cannot be opened or written.
        pickle.PicklingError: If the classifier cannot be serialized.
    """
    with path.open("xb") as stream:
        pickle.dump(
            model,
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
