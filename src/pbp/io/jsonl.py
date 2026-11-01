"""Read battle records."""

import json
from pathlib import Path
from typing import TypedDict, cast

from tqdm import tqdm


class PokemonDetails(TypedDict):
    """Pokémon base statistics."""

    base_hp: int
    base_atk: int
    base_def: int
    base_spa: int
    base_spd: int
    base_spe: int


class PokemonState(TypedDict):
    """Observed Pokémon state."""

    name: str
    hp_pct: float
    status: str


class Turn(TypedDict):
    """Observed turn states."""

    p1_pokemon_state: PokemonState
    p2_pokemon_state: PokemonState


class Battle(TypedDict):
    """Battle record."""

    battle_id: int
    p1_team_details: list[PokemonDetails]
    p2_lead_details: PokemonDetails
    battle_timeline: list[Turn]


class TrainingBattle(Battle):
    """Labeled battle record."""

    player_won: int


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
