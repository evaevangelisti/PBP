"""Describe battle records."""

from typing import TypedDict


class PokemonDetails(TypedDict):
    """Pokémon details."""

    name: str
    level: int
    types: list[str]
    base_hp: int
    base_atk: int
    base_def: int
    base_spa: int
    base_spd: int
    base_spe: int


class TeamPokemonDetails(PokemonDetails):
    """Team Pokémon details."""

    item: str


StatBoosts = TypedDict(
    "StatBoosts",
    {
        "atk": int,
        "def": int,
        "spa": int,
        "spd": int,
        "spe": int,
    },
)
"""In-battle stat boosts."""


class PokemonState(TypedDict):
    """Observed Pokémon state."""

    name: str
    hp_pct: float
    status: str
    effects: list[str]
    boosts: StatBoosts


class MoveDetails(TypedDict):
    """Move details."""

    name: str
    type: str
    category: str
    base_power: int
    accuracy: float
    priority: int


class Turn(TypedDict):
    """Observed turn details."""

    turn: int
    p1_pokemon_state: PokemonState
    p1_move_details: MoveDetails | None
    p2_pokemon_state: PokemonState
    p2_move_details: MoveDetails | None


class Battle(TypedDict):
    """Battle record."""

    battle_id: int
    p1_team_details: list[TeamPokemonDetails]
    p2_lead_details: PokemonDetails
    battle_timeline: list[Turn]


class TrainingBattle(Battle):
    """Labeled battle record."""

    player_won: int
