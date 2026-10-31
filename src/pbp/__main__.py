"""Train a classifier and write test predictions."""

import argparse
from pathlib import Path
from typing import cast

from sklearn.linear_model import LogisticRegression

from .features import extract_features
from .io import read_battles, save_model, write_predictions
from .io.jsonl import TrainingBattle
from .prediction import predict
from .training import train_model


def main(
    training_path: Path,
    test_path: Path,
    output_directory: Path,
) -> None:
    """Train a classifier, save it, and write test predictions.

    Args:
        training_path: Path to labeled training battles in JSON Lines format.
        test_path: Path to unlabeled test battles in JSON Lines format.
        output_directory: Directory created for the model and prediction files.

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

    training_features = extract_features(training_battles)
    test_features = extract_features(test_battles)

    targets = [
        int(cast(TrainingBattle, battle)["player_won"]) for battle in training_battles
    ]

    model = train_model(
        LogisticRegression(max_iter=1000),
        training_features,
        targets,
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    save_model(
        output_directory / "model.pkl",
        model,
    )

    predictions = predict(model, test_features)

    write_predictions(
        output_directory / "predictions.csv",
        [battle["battle_id"] for battle in test_battles],
        predictions,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a classifier and predict battle outcomes.",
    )

    parser.add_argument("training_path", type=Path)
    parser.add_argument("test_path", type=Path)

    parser.add_argument("output_directory", type=Path)

    arguments = parser.parse_args()

    main(
        cast(Path, arguments.training_path),
        cast(Path, arguments.test_path),
        cast(Path, arguments.output_directory),
    )
