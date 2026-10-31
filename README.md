# Pokémon Battles Prediction

Predict Generation 2 OU Pokémon battle outcomes from the first 30 turns.

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

```sh
uv run python -m pbp data/train.jsonl data/test.jsonl outputs/
```
