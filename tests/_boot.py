"""Adds the project root to sys.path so tests import `samanvay` correctly
regardless of how they are invoked (unittest discover, a plain `python3
tests/test_x.py`, or a future pytest run). Import this first in every test
module: `from . import _boot  # noqa: F401`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
