"""Save trained models."""

import pickle
from pathlib import Path

from ..training import Classifier


def save_model(
    path: Path,
    model: Classifier,
) -> None:
    """Save a model.

    Args:
        path: Output file.
        model: Classifier to save.

    Raises:
        OSError: If writing fails.
        pickle.PicklingError: If serialization fails.
    """
    with path.open("wb") as stream:
        pickle.dump(
            model,
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
