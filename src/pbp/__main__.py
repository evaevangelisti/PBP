"""Run the prediction pipeline."""

import argparse
import logging
import pickle
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
    """Train a classifier and predict outcomes.

    Args:
        training_path: Training file.
        test_path: Test file.
        output_directory: Output directory.

    Raises:
        OSError: If file access fails.
        ValueError: If the data is unusable.
        pickle.PicklingError: If serialization fails.
    """
    training_records, _ = read_battles(training_path)
    training_battles = [
        cast(TrainingBattle, battle)
        for battle in training_records
        if battle.get("player_won") in (0, 1)
    ]

    if not training_battles:
        raise ValueError("No training battles with usable targets were read.")

    test_battles, skipped_test_records = read_battles(test_path)

    if not test_battles:
        raise ValueError("No test battles were read.")

    training_features = extract_features(
        training_battles,
        description="Extracting training features",
    )

    if training_features.empty:
        raise ValueError("No training battles remain after feature extraction.")

    test_features = extract_features(
        test_battles,
        description="Extracting test features",
    )

    if test_features.empty:
        raise ValueError("No test battles remain after feature extraction.")

    skipped_test_records += len(test_battles) - len(test_features)

    if skipped_test_records:
        _logger.warning(
            "Skipped %d test records; the prediction file will be incomplete.",
            skipped_test_records,
        )

    targets = [
        int(training_battles[cast(int, battle_index)]["player_won"])
        for battle_index in training_features.index
    ]

    model = LogisticRegression(max_iter=1000)

    _logger.info(
        "Training %s on %d battles with %d features.",
        type(model).__name__,
        len(training_features),
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

    _logger.info("Predicting %d test battles.", len(test_features))

    predictions = predict(model, test_features)

    predictions_path = output_directory / "predictions.csv"
    write_predictions(
        predictions_path,
        [
            test_battles[cast(int, battle_index)]["battle_id"]
            for battle_index in test_features.index
        ],
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

    try:
        main(
            cast(Path, arguments.training_path),
            cast(Path, arguments.test_path),
            cast(Path, arguments.output_directory),
        )
    except OSError as error:
        if error.filename is None:
            _logger.error("%s", error.strerror)
        else:
            _logger.error("%s (%s)", error.strerror, error.filename)

        raise SystemExit(1) from error
    except (ValueError, pickle.PicklingError) as error:
        _logger.error("%s", error)

        raise SystemExit(1) from error
