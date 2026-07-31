"""QaraqalpaqMind - an open LLM stack for the Karakalpak (qaraqalpaq, ISO 639-3 `kaa`) language."""

from __future__ import annotations

from .common.env import load_env

# Must run before any submodule reads the environment: `common.paths` resolves
# QM_DATA_DIR at import time, and the source registry expands ${QM_USER_AGENT}
# when parsed. Importing the package is the only point early enough.
load_env()

__version__ = "0.1.0"
__all__ = ["__version__"]
