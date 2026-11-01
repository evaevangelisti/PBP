"""Configure Sphinx."""

from importlib import metadata

# Project information.
project: str = "Pokémon Battles Prediction"
author: str = "Eva"
project_copyright: str = "2026, Eva"
release: str = metadata.version("pbp")

# Documentation sources and extensions.
extensions: list[str] = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Docstring rendering.
napoleon_google_docstring: bool = True
napoleon_numpy_docstring: bool = False
napoleon_use_ivar: bool = True
autodoc_member_order: str = "bysource"
autodoc_typehints: str = "description"

# HTML presentation.
html_theme: str = "shibuya"
