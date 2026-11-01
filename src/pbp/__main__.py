"""Train a classifier and write test predictions."""

import argparse
import logging
from pathlib import Path
from typing import cast

from sklearn.linear_model import LogisticRegression

from .features import extract_features
from .io import read_battles, save_model, write_predictions
from .io.jsonl import TrainingBattle
from .prediction import predict
from .training import train_model

_logger: logging.Logger = logging.getLogger(__name__)


def main(
    training_path: Path,
    test_path: Path,
    output_directory: Path,
) -> None:
    """Train a classifier, save it, and write test predictions.

    Args:
        training_path: Path to labeled training battles in JSON Lines format.
        test_path: Path to unlabeled test battles in JSON Lines format.
        output_directory: Destination directory for the model and predictions.

    Raises:
        FileExistsError: If either destination file already exists.
        OSError: If an input cannot be read or an output cannot be written.
        json.JSONDecodeError: If an input line contains invalid JSON.
        KeyError: If a battle lacks a required field.
        ValueError: If training or prediction receives unsupported data.
        pickle.PicklingError: If the fitted classifier cannot be serialized.
    """
    training_battles = read_battles(training_path)

    test_battles = read_battles(test_path)

    training_features = extract_features(
        training_battles,
        description="Extracting training features",
    )

    test_features = extract_features(
        test_battles,
        description="Extracting test features",
    )

    targets = [
        int(cast(TrainingBattle, battle)["player_won"]) for battle in training_battles
    ]

    model = LogisticRegression(max_iter=1000)

    _logger.info(
        "Training %s on %d battles with %d features.",
        type(model).__name__,
        len(training_battles),
        training_features.shape[1],
    )

    model = train_model(
        model,
        training_features,
        targets,
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    model_path = output_directory / "model.pkl"
    save_model(model_path, model)

    _logger.info("Model saved to %s.", model_path)

    _logger.info("Predicting %d test battles.", len(test_battles))
    predictions = predict(model, test_features)

    predictions_path = output_directory / "predictions.csv"
    write_predictions(
        predictions_path,
        [battle["battle_id"] for battle in test_battles],
        predictions,
    )

    _logger.info("Predictions saved to %s.", predictions_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a classifier and predict battle outcomes",
    )

    parser.add_argument("training_path", type=Path)
    parser.add_argument("test_path", type=Path)

    parser.add_argument(
        "--output-directory",
        "-o",
        default=Path(),
        type=Path,
        help="Directory for the model and predictions (default: current directory)",
    )

    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    main(
        cast(Path, arguments.training_path),
        cast(Path, arguments.test_path),
        cast(Path, arguments.output_directory),
    )
