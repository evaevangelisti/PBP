# Pokémon Battles Prediction

Predict Generation 2 OU Pokémon battle outcomes from the first 30 turns.

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

```sh
uv run python -m pbp data/raw/train.jsonl data/raw/test.jsonl
```

| Parameter | Description |
| --- | --- |
| `training_path` | Labeled training battles in JSON Lines format. |
| `test_path` | Unlabeled test battles in JSON Lines format. |
| `--output-directory`, `-o` | Destination for `model.pkl` and `predictions.csv` (default: `.`). |
