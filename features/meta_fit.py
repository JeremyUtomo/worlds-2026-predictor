"""Meta fit: does the team play what is strong right now, and does its style suit
how games are being won?

* champ_fit     - average meta score of each player's recent champion pool
* meta_mastery  - same, but weighted by how well the player wins on those champs
* depth         - meta-tier champions each player has real games on (ban resilience)
* ban_vuln      - how much each player leans on one meta champion
* style_fit     - early-game strength x how decisive early leads are in this meta,
                  plus comeback ability x how often games are thrown back
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _zero_pair() -> list:
    return [0.0, 0.0]


class MetaFit:
    def __init__(self, cfg: dict, meta, picks: pd.DataFrame):
        self.meta = meta
        self.pool_window = pd.Timedelta(days=cfg["meta"]["pool_days"])
        self.by_player: dict[str, tuple] = {}
        for pid, g in picks.groupby("player"):
            self.by_player[pid] = (g.date.values, g.champ.values, g.win.values.astype(float))
        self.comeback = defaultdict(_zero_pair)   # team -> [wins when behind, times behind]

    def update_game(self, team: str, behind_at_15: bool, won: bool) -> None:
        if behind_at_15:
            c = self.comeback[team]
            c[0] += won
            c[1] += 1

    def _player(self, pid: str, date, sd: dict, top) -> tuple:
        if pid not in self.by_player:
            return 0.0, 0.0, 0.0, 0.0
        d, c, w = self.by_player[pid]
        hi = np.searchsorted(d, np.datetime64(date.tz_convert(None)))
        lo = np.searchsorted(d, np.datetime64((date - self.pool_window).tz_convert(None)))
        if hi <= lo:
            return 0.0, 0.0, 0.0, 0.0
        names, inv = np.unique(c[lo:hi], return_inverse=True)
        count = np.bincount(inv).astype(float)
        wsum = np.bincount(inv, weights=w[lo:hi])
        total = count.sum()
        share = count / total
        ms = np.array([sd.get(n, 0.0) for n in names])
        wr = (wsum + 1.5) / (count + 3)
        fit = float((share * ms).sum())
        mastery = float((share * ms * (wr - 0.5) * 2).sum())
        is_top = np.array([n in top for n in names])
        meta_counts = count[is_top]
        depth = min(int((meta_counts >= 3).sum()), 5) / 5
        vuln = float(meta_counts.max() / total) if len(meta_counts) else 0.0
        return fit, mastery, depth, vuln

    def team(self, team: str, roster: list, date, gd15_form: float) -> dict:
        day = self.meta.day(date)
        scores = self.meta.scores_dict(day)
        top = self.meta.top_set(day)
        vals = np.array([self._player(p, date, scores, top) for p in roster if p]) \
            if roster else np.zeros((0, 4))
        if len(vals) == 0:
            vals = np.zeros((1, 4))
        style = self.meta.style(day)
        w, n = self.comeback[team]
        comeback = (w + 0.25 * 6) / (n + 6)
        early = style["early_conv"]
        style_fit = (gd15_form / 1000) * (early - 0.5) * 2 + (comeback - 0.25) * (1 - early) * 2
        m = vals.mean(axis=0)
        return {"champ_fit": m[0], "meta_mastery": m[1], "depth": m[2], "ban_vuln": m[3],
                "style_fit": style_fit}

    def features(self, a, b, roster_a, roster_b, date, gd15_a, gd15_b) -> dict:
        fa = self.team(a, roster_a, date, gd15_a)
        fb = self.team(b, roster_b, date, gd15_b)
        return {k: fa[k] - fb[k] for k in fa}
