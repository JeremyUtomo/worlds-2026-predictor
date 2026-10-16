"""Copy the seed snapshot into the data directory on a fresh install.

The seed (series results, ~6,500 games with player stats, backtest, calibration)
means a new server starts in seconds instead of downloading two years of games.
Files that already exist are never overwritten.
"""
from __future__ import annotations

import shutil

import config

SEED_FILES = ["series.csv", "games.pkl", "backtest.pkl", "calibration.json"]


def seed() -> list[str]:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in SEED_FILES:
        src, dst = config.SEED_DIR / name, config.DATA_DIR / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


if __name__ == "__main__":
    print("seeded:", seed() or "nothing (data already present)")
