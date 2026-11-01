"""Extract numerical features from observed battle records."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .io.jsonl import Battle


def extract_features(
    battles: Sequence[Battle],
    *,
    description: str = "Extracting features",
) -> pd.DataFrame:
    """Extract base statistics and observed knockout counts.

    Args:
        battles: Battle records in the desired row order.
        description: Label displayed beside the progress bar.

    Returns:
        A numerical feature table preserving the input battle order.

    Raises:
        KeyError: If a required field is missing.
    """
    rows: list[dict[str, float]] = []

    for battle in tqdm(battles, desc=description, unit="battle"):
        team = battle["p1_team_details"]
        opponent_lead = battle["p2_lead_details"]

        timeline = battle["battle_timeline"]

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

        rows.append(
            {
                "p1_mean_hp": float(np.mean([pokemon["base_hp"] for pokemon in team])),
                "p1_mean_atk": float(
                    np.mean([pokemon["base_atk"] for pokemon in team])
                ),
                "p1_mean_def": float(
                    np.mean([pokemon["base_def"] for pokemon in team])
                ),
                "p1_mean_spa": float(
                    np.mean([pokemon["base_spa"] for pokemon in team])
                ),
                "p1_mean_spd": float(
                    np.mean([pokemon["base_spd"] for pokemon in team])
                ),
                "p1_mean_spe": float(
                    np.mean([pokemon["base_spe"] for pokemon in team])
                ),
                "p2_lead_hp": opponent_lead["base_hp"],
                "p2_lead_atk": opponent_lead["base_atk"],
                "p2_lead_def": opponent_lead["base_def"],
                "p2_lead_spa": opponent_lead["base_spa"],
                "p2_lead_spd": opponent_lead["base_spd"],
                "p2_lead_spe": opponent_lead["base_spe"],
                "p1_pokemon_fainted": float(len(player_1_fainted)),
                "p2_pokemon_fainted": float(len(player_2_fainted)),
            },
        )

    return pd.DataFrame(rows)
