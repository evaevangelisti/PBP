"""Read type interactions and write prediction tables."""

import csv
from collections.abc import Sequence
from pathlib import Path


def read_type_chart(
    path: Path,
) -> dict[tuple[str, str], float]:
    """Read the type multipliers explicitly listed in a CSV file.

    Args:
        path: Path to the Generation 2 type chart.

    Returns:
        A mapping from attacking and defending type pairs to listed multipliers.

    Raises:
        OSError: If the input file cannot be opened or read.
        KeyError: If a required column is missing.
        ValueError: If a multiplier cannot be converted to a float.
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
    """Write battle identifiers and predictions, replacing an existing CSV file.

    Args:
        path: Destination path for the CSV file.
        battle_ids: Battle identifiers in prediction order.
        predictions: Binary outcomes aligned with the battle identifiers.

    Raises:
        OSError: If the destination file cannot be opened or written.
        ValueError: If sequence lengths differ, possibly after writing some rows.
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
