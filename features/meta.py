"""Meta: how strong each champion is right now, and what kind of game wins.

Champion meta score = pick share x (1 + 2 x (shrunk win rate - 0.5)) over a trailing
window of pro games, with international games weighted up and a buff/nerf
adjustment for a new patch that fades as games on that patch accumulate.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import yaml

import config


def picks_table(games: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (game, champion pick)."""
    rows = []
    for g in games.itertuples():
        for side in ("blue", "red"):
            won = g.winner == side
            team = getattr(g, side)
            for champ, player, role in zip(getattr(g, f"{side}_champs"),
                                           getattr(g, f"{side}_players"),
                                           getattr(g, f"{side}_roles")):
                rows.append((g.date, g.game_id, g.league, g.patch, team, player, role, champ, won))
    df = pd.DataFrame(rows, columns=["date", "game_id", "league", "patch", "team", "player",
                                     "role", "champ", "win"])
    df["intl"] = df.league.isin(config.INTL_LEAGUES)
    return df.sort_values("date").reset_index(drop=True)


def _vkey(v) -> int:
    try:
        major, minor = str(v).split(".")[:2]
        return int(major) * 100 + int(minor)
    except ValueError:
        return 0


def load_patch_changes() -> list[dict]:
    if not config.PATCH_FILE.exists():
        return []
    return yaml.safe_load(config.PATCH_FILE.read_text()).get("patches", [])


class Meta:
    def __init__(self, cfg: dict, picks: pd.DataFrame, games: pd.DataFrame):
        m = cfg["meta"]
        self.window = pd.Timedelta(days=m["window_days"])
        self.intl_w = m["current_event_weight"]
        self.patch_strength = m["patch_strength"]
        self.top_n = m["top_n"]
        self.picks = picks
        self.p_dates = picks.date.values
        self.games = games
        self.g_dates = games.date.values if len(games) else np.array([], dtype="datetime64[ns]")
        self.patches = load_patch_changes()

    def _slice(self, df, dates, date):
        lo = np.searchsorted(dates, np.datetime64((date - self.window).tz_convert(None)))
        hi = np.searchsorted(dates, np.datetime64(date.tz_convert(None)))
        return df.iloc[lo:hi]

    @lru_cache(maxsize=4096)
    def scores(self, day: pd.Timestamp) -> pd.Series:
        """Champion -> meta score in [0, 1] as of the start of ``day``."""
        p = self._slice(self.picks, self.p_dates, day)
        if p.empty:
            return pd.Series(dtype=float)
        w = np.where(p.intl, self.intl_w, 1.0)
        n_games = (pd.Series(w, index=p.index).groupby(p.game_id).first()).sum()
        grp = pd.DataFrame({"champ": p.champ, "w": w, "ww": w * p.win}).groupby("champ").sum()
        pick_share = grp.w / n_games
        wr = (grp.ww + 5) / (grp.w + 10)
        score = pick_share * (1 + 2 * (wr - 0.5))
        score = self._patch_adjust(score, day, p)
        return (score / score.max()).clip(lower=0).sort_values(ascending=False)

    def _patch_adjust(self, score: pd.Series, day, p) -> pd.Series:
        for patch in self.patches:
            live = pd.Timestamp(patch["live"], tz="UTC")
            if day < live:
                continue
            n_new = p[p.patch.map(_vkey) >= _vkey(patch["game_version"])].game_id.nunique()
            fade = 1 / (1 + n_new / 150)
            for champ, mag in (patch.get("changes") or {}).items():
                base = score.get(champ, score.median() * 0.5 if mag > 0 else 0.0)
                score.loc[champ] = base * (1 + self.patch_strength * mag * fade)
        return score

    @lru_cache(maxsize=4096)
    def scores_dict(self, day: pd.Timestamp) -> dict:
        return self.scores(day).to_dict()

    @lru_cache(maxsize=4096)
    def top_set(self, day: pd.Timestamp) -> frozenset:
        return frozenset(self.top(day))

    def top(self, day) -> list[str]:
        return list(self.scores(day).index[: self.top_n])

    @lru_cache(maxsize=4096)
    def style(self, day: pd.Timestamp) -> dict:
        """How decisive early leads are in the current meta."""
        g = self._slice(self.games, self.g_dates, day)
        g = g[g.gd15.notna()] if "gd15" in g else g.iloc[0:0]
        if len(g) < 30:
            return {"early_conv": 0.72, "minutes": 32.0}
        big = g[g.gd15.abs() > 1500]
        conv = np.mean((big.gd15 > 0) == (big.winner == "blue")) if len(big) else 0.72
        return {"early_conv": float(conv), "minutes": float(g.minutes.mean())}

    def day(self, date) -> pd.Timestamp:
        return date.floor("D")
