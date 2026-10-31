# Usage

```{include} ../../README.md
:start-after: "## Usage"
```

| Argument | Description |
| --- | --- |
| `training_path` | Labeled training battles in JSON Lines format. |
| `test_path` | Unlabeled test battles in JSON Lines format. |
| `output_directory` | Destination for `model.pkl` and `predictions.csv`. |

Feature extraction is not implemented yet, so the command currently stops before
training. Once implemented, the pipeline trains on all supplied training battles
and writes the model and test predictions without replacing existing files.
