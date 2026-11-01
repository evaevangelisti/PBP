"""Read and write CSV files."""

import csv
from collections.abc import Sequence
from pathlib import Path


def read_type_chart(
    path: Path,
) -> dict[tuple[str, str], float]:
    """Read the type chart.

    Args:
        path: Input file.

    Returns:
        Multipliers by attacking and defending type.

    Raises:
        OSError: If the file cannot be read.
        KeyError: If a required column is missing.
        ValueError: If a multiplier is invalid.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        return {
            (
                row["attacking_type"],
                row["defending_type"],
            ): float(row["multiplier"])
            for row in csv.DictReader(stream)
        }


def write_predictions(
    path: Path,
    battle_ids: Sequence[int],
    predictions: Sequence[int],
) -> None:
    """Write predictions.

    Args:
        path: Output file.
        battle_ids: Battle identifiers in row order.
        predictions: Binary predictions in matching order.

    Raises:
        OSError: If writing fails.
        ValueError: If identifier and prediction counts differ.
    """
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)

        writer.writerow(
            (
                "battle_id",
                "player_won",
            ),
        )

        writer.writerows(zip(battle_ids, predictions, strict=True))
