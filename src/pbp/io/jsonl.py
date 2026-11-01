"""Read battle records."""

import json
from pathlib import Path
from typing import cast

from tqdm import tqdm

from ..models import Battle


def read_battles(
    path: Path,
) -> tuple[list[Battle], int]:
    """Read battle records.

    Args:
        path: Input file.

    Returns:
        Parsed battles and number of skipped lines.

    Raises:
        OSError: If the file cannot be read.
    """
    battles: list[Battle] = []
    skipped_count = 0

    with path.open("rb") as stream:
        for line in tqdm(stream, desc=str(path), unit=" battle"):
            try:
                battle = cast(Battle, json.loads(line))
                _ = battle["battle_id"]
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
                skipped_count += 1

                continue

            battles.append(battle)

    return battles, skipped_count
