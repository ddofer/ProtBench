"""Shared pytest config: ensure ``plm/bench`` is on ``sys.path`` for tests."""

import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent.parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))


def pytest_configure(config):
    # Register custom marker so ``-m slow`` works without warnings, even if
    # the parent pyproject.toml doesn't declare it.
    config.addinivalue_line(
        "markers", "slow: heavyweight tests that hit the network / download models"
    )
