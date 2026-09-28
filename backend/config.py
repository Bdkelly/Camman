"""Writable application paths, independent of the installed package location."""

import os
from pathlib import Path


def models_directory():
    return Path(
        os.environ.get("CAMMAN_MODELS_DIR", Path.home() / ".camman" / "models")
    ).expanduser()
