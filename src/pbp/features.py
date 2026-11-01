"""Extract battle features."""

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
from tqdm import tqdm

from .models import Battle, PokemonDetails, Turn

_BASE_STATISTICS: Final = (
    "base_hp",
    "base_atk",
    "base_def",
    "base_spa",
    "base_spd",
    "base_spe",
)


def _extract_team_features(
    team: Sequence[PokemonDetails],
) -> dict[str, float]:
    """Extract team features.

    Args:
        team: Player 1 team.

    Returns:
        Team features.
    """
    statistics = np.array(
        [[pokemon[statistic] for statistic in _BASE_STATISTICS] for pokemon in team],
    )

    mean_statistics = np.mean(statistics, axis=0)

    return {
        f"p1_mean_{statistic.removeprefix('base_')}": float(value)
        for statistic, value in zip(_BASE_STATISTICS, mean_statistics, strict=True)
    }


def _extract_lead_features(
    lead: PokemonDetails,
) -> dict[str, float]:
    """Extract lead features.

    Args:
        lead: Opponent lead.

    Returns:
        Lead features.
    """
    return {
        f"p2_lead_{statistic.removeprefix('base_')}": lead[statistic]
        for statistic in _BASE_STATISTICS
    }


def _extract_timeline_features(
    timeline: Sequence[Turn],
) -> dict[str, float]:
    """Extract timeline features.

    Args:
        timeline: Battle turns.

    Returns:
        Timeline features.
    """
    player_1_fainted = {
        turn["p1_pokemon_state"]["name"]
        for turn in timeline
        if turn["p1_pokemon_state"]["status"] == "fnt"
    }

    player_2_fainted = {
        turn["p2_pokemon_state"]["name"]
        for turn in timeline
        if turn["p2_pokemon_state"]["status"] == "fnt"
    }

    return {
        "p1_pokemon_fainted": float(len(player_1_fainted)),
        "p2_pokemon_fainted": float(len(player_2_fainted)),
    }


def _extract_battle_features(
    battle: Battle,
) -> dict[str, float]:
    """Extract battle features.

    Args:
        battle: Input battle.

    Returns:
        Battle features.
    """
    return {
        **_extract_team_features(battle["p1_team_details"]),
        **_extract_lead_features(battle["p2_lead_details"]),
        **_extract_timeline_features(battle["battle_timeline"]),
    }


def extract_features(
    battles: Sequence[Battle],
    *,
    description: str = "Extracting features",
) -> pd.DataFrame:
    """Extract features.

    Args:
        battles: Input battles.
        description: Progress bar label.

    Returns:
        Features indexed by input position.
    """
    rows: list[dict[str, float]] = []
    retained_indices: list[int] = []

    for battle_index, battle in enumerate(
        tqdm(
            battles,
            desc=description,
            unit="battle",
        ),
    ):
        try:
            if not battle["p1_team_details"] or not battle["battle_timeline"]:
                continue

            with np.errstate(divide="raise", over="raise", invalid="raise"):
                row = _extract_battle_features(battle)

                if not np.isfinite(list(row.values())).all():
                    continue
        except (KeyError, TypeError, ValueError, FloatingPointError, OverflowError):
            continue

        rows.append(row)
        retained_indices.append(battle_index)

    return pd.DataFrame(rows, index=retained_indices)
