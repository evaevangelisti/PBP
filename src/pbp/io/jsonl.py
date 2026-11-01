"""Read battle records from JSON Lines files."""

import json
from pathlib import Path
from typing import TypedDict, cast

from tqdm import tqdm


class PokemonDetails(TypedDict):
    """Describe the base statistics used for feature extraction."""

    base_hp: int
    base_atk: int
    base_def: int
    base_spa: int
    base_spd: int
    base_spe: int


class PokemonState(TypedDict):
    """Describe the observed state of an active Pokémon."""

    name: str
    hp_pct: float
    status: str


class Turn(TypedDict):
    """Describe the states observed during a turn."""

    p1_pokemon_state: PokemonState
    p2_pokemon_state: PokemonState


class Battle(TypedDict):
    """Describe the battle fields used for feature extraction."""

    battle_id: int
    p1_team_details: list[PokemonDetails]
    p2_lead_details: PokemonDetails
    battle_timeline: list[Turn]


class TrainingBattle(Battle):
    """Include the outcome available in training records."""

    player_won: int


def read_battles(
    path: Path,
) -> tuple[list[Battle], int]:
    """Read battle records and skip malformed JSON Lines records.

    Args:
        path: Path to the input JSON Lines file.

    Returns:
        A tuple containing parsed battle records and the number of skipped lines.

    Raises:
        OSError: If the input file cannot be opened or read.
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
