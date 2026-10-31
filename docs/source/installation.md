# Installation

```{include} ../../README.md
:start-after: "## Installation"
:end-before: "## Usage"
```

## Documentation

Build the HTML documentation with warnings treated as errors:

```sh
uv run --group docs sphinx-build -W --keep-going docs/source docs/build/html
```

Open `docs/build/html/index.html` after the build.
