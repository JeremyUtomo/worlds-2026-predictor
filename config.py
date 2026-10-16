"""Paths, league list, Worlds 2026 field and default model settings."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# runtime state (downloads, logs, model state); /data inside the container
DATA_DIR = Path(os.environ.get("WORLDS_DATA", ROOT / "data"))
SETTINGS_DIR = ROOT / "settings"      # hand-edited inputs, versioned in git
SEED_DIR = ROOT / "seed"              # data snapshot used to bootstrap a fresh server
CACHE_DIR = DATA_DIR / "cache"
# "server": a separate updater process refreshes the model; the web app only reads
MODE = os.environ.get("WORLDS_MODE", "local")
CONFIG_FILE = ROOT / "model_config.json"
REPORT_FILE = ROOT / "research_report.md"
PREDICTION_LOG = DATA_DIR / "prediction_log.csv"
FORECAST_LOG = DATA_DIR / "forecast_log.csv"
CALIBRATION_FILE = DATA_DIR / "calibration.json"
PATCH_FILE = SETTINGS_DIR / "patch_changes.yaml"
BRACKET_FILE = SETTINGS_DIR / "bracket.yaml"
ODDS_FILE = SETTINGS_DIR / "market_odds.yaml"

# Public key used by the lolesports.com web app itself.
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
API_BASE = "https://esports-api.lolesports.com/persisted/gw"
FEED_BASE = "https://feed.lolesports.com/livestats/v1"

# league slug -> (league id, region bucket). LTA / PCS / VCS / LJL / LCO map onto
# the region that later fielded the team (LCS, CBLOL, LCP).
LEAGUES = {
    "worlds": ("98767975604431411", "INTL"),
    "msi": ("98767991325878492", "INTL"),
    "first_stand": ("113464388705111224", "INTL"),
    "ewc_lol": ("116838530616006090", "INTL"),
    "lck": ("98767991310872058", "LCK"),
    "lpl": ("98767991314006698", "LPL"),
    "lec": ("98767991302996019", "LEC"),
    "lcs": ("98767991299243165", "LCS"),
    "lta_n": ("113470291645289904", "LCS"),
    "cblol-brazil": ("98767991332355509", "CBLOL"),
    "lta_s": ("113475181634818701", "CBLOL"),
    "lta_cross": ("113475149040947852", "INTL"),
    "lcp": ("113476371197627891", "LCP"),
    "pcs": ("104366947889790212", "LCP"),
    "vcs": ("107213827295848783", "LCP"),
    "ljl-japan": ("98767991349978712", "LCP"),
}
INTL_LEAGUES = {s for s, (_, r) in LEAGUES.items() if r == "INTL"}

SERIES_SINCE = "2023-06-01"   # results for Elo warm-up
GAMES_SINCE = "2024-08-01"    # per-game champion / gold stats (meta features)

# Worlds 2026 field: code -> region. Codes are lolesports team codes.
WORLDS_2026 = {
    "GEN": "LCK", "HLE": "LCK", "T1": "LCK", "DK": "LCK",
    "AL": "LPL", "BLG": "LPL", "TES": "LPL", "IG": "LPL",
    "G2": "LEC", "MKOI": "LEC", "KC": "LEC",
    "TLAW": "LCS", "LYON": "LCS", "C9": "LCS",
    "TSW": "LCP", "CFO": "LCP", "MVK": "LCP",
    "LOS": "CBLOL", "FUR": "CBLOL",
}
PLAY_IN = ["KC", "C9", "MVK", "FUR"]

# Pick'Ems scoring (2026 rules as published; bonus is an estimate - edit if needed).
PICKEMS = {
    "swiss_slots": {"3-0": 2, "3-1": 3, "3-2": 3},
    "swiss_advance_pts": 10,
    "swiss_exact_bonus": 5,
    "ko_pts": {"QF": 10, "SF": 20, "F": 80},
}

DEFAULT_CONFIG = {
    "elo": {
        "k_long": 24.0,          # long-run team rating
        "k_fast": 48.0,          # form rating
        "fast_half_life_days": 60.0,
        "k_region": 12.0,        # region rating, international games only
        "season_regress": 0.75,  # pull toward region mean each January
    },
    "form": {"ewma_games": 12},
    "history": {"h2h_prior_games": 6, "intl_prior_games": 8},
    "roster": {"blend": True, "k_player": 12.0, "rookie_start": 1.0, "continuity_days": 365, "continuity_half_life_days": 90},
    "player_form": {"ewma_games": 10},
    "meta": {
        "window_days": 45,
        "current_event_weight": 3.0,
        "patch_strength": 0.25,
        "top_n": 25,
        "pool_days": 120,
    },
    "model": {
        "groups": ["form", "history", "roster", "meta_fit"],
        "l2_C": 1.0,
        "intl_weight": 3.0,
        "train_since": "2024-01-01",
    },
    "calibration": {"min_events": 6, "shrink": 5.0},
}


def load_config() -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        saved = json.loads(CONFIG_FILE.read_text())
        for k, v in saved.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg
