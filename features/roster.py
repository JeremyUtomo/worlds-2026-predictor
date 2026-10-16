"""Roster group: players, not team names.

A team's record only means something if the same players produced it. So:

* every player has their own rating, updated from the games they play and
  carried with them when they change team (a rookie starts at the rating of
  the team they join, shrunk toward league average by ``rookie_start``);
* ``continuity`` = how much of today's five-man roster played the team's games
  over the last year (recency weighted). Same five -> 1.0; four new faces -> ~0.2;
* ``elo_roster`` blends the two: continuity x team rating + (1 - continuity) x
  average player rating, so a rebuilt team is judged on its players' records;
* ``roster_gap`` = (1 - continuity) x (player rating - team rating): zero for a
  stable roster, and a correction toward the new players' level after changes;
* the history group's team-level stats (international win rate, head-to-head)
  are multiplied by continuity, so they fade when the roster that earned them
  is gone.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


class Roster:
    def __init__(self, cfg: dict, history):
        r = cfg["roster"]
        self.k = r["k_player"]
        self.rookie = r["rookie_start"]
        self.window = pd.Timedelta(days=r["continuity_days"])
        self.half_life = r["continuity_half_life_days"]
        self.h = history
        self.player_off: dict[str, float] = {}
        self.lineups: dict[str, list] = defaultdict(list)   # team -> [(date, frozenset)]

    def _off(self, p: str, team: str) -> float:
        if p not in self.player_off:
            self.player_off[p] = self.rookie * self.h.team_off[team]
        return self.player_off[p]

    def player_rating(self, team: str, roster: list) -> float:
        base = self.h.region_rating.get(self.h.region_of(team), 0.0)
        if not roster:
            return base + self.h.team_off[team]
        return base + float(np.mean([self._off(p, team) for p in roster]))

    def update_game(self, date, blue: str, red: str, bp: list, rp: list, blue_won: bool) -> None:
        if len(bp) != 5 or len(rp) != 5:
            return
        e = 1 / (1 + 10 ** ((self.player_rating(red, rp) - self.player_rating(blue, bp)) / 400))
        s = (1.0 if blue_won else 0.0) - e
        for p in bp:
            self.player_off[p] = self._off(p, blue) + self.k * s
        for p in rp:
            self.player_off[p] = self._off(p, red) - self.k * s
        self.lineups[blue].append((date, frozenset(bp)))
        self.lineups[red].append((date, frozenset(rp)))

    def new_year(self, regress: float) -> None:
        for p in self.player_off:
            self.player_off[p] *= regress

    def continuity(self, team: str, roster: list, date) -> float:
        games = [(d, l) for d, l in self.lineups.get(team, []) if date - d <= self.window]
        if not games or not roster:
            return 0.0
        cur = set(roster)
        w = np.array([0.5 ** ((date - d).days / self.half_life) for d, _ in games])
        overlap = np.array([len(cur & l) / 5 for _, l in games])
        return float((w * overlap).sum() / w.sum())

    def team(self, team: str, roster: list, date) -> dict:
        c = self.continuity(team, roster, date)
        pr = self.player_rating(team, roster)
        tr = self.h.rating(team)
        return {"continuity": c, "elo_players": pr, "elo_roster": c * tr + (1 - c) * pr,
                "roster_gap": (1 - c) * (pr - tr)}

    def features(self, a, b, ra, rb, date) -> tuple[dict, float, float]:
        fa, fb = self.team(a, ra, date), self.team(b, rb, date)
        return ({"elo_players": (fa["elo_players"] - fb["elo_players"]) / 400,
                 "elo_roster": (fa["elo_roster"] - fb["elo_roster"]) / 400,
                 "continuity": fa["continuity"] - fb["continuity"],
                 "roster_gap": (fa["roster_gap"] - fb["roster_gap"]) / 400},
                fa["continuity"], fb["continuity"])
