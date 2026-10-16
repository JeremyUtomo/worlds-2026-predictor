"""History group: long-run team rating with a region layer, international record,
head-to-head and Worlds experience of the current roster."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

TWO_YEARS = pd.Timedelta(days=730)
THREE_YEARS = pd.Timedelta(days=1095)

REGION_PRIOR = {"LCK": 120.0, "LPL": 110.0, "LEC": 0.0, "LCS": -60.0,
                "LCP": -60.0, "CBLOL": -120.0}


class History:
    def __init__(self, cfg: dict):
        self.k_long = cfg["elo"]["k_long"]
        self.k_region = cfg["elo"]["k_region"]
        self.regress = cfg["elo"]["season_regress"]
        self.h2h_prior = cfg["history"]["h2h_prior_games"]
        self.intl_prior = cfg["history"]["intl_prior_games"]
        self.region_rating = dict(REGION_PRIOR)
        self.team_off: dict[str, float] = defaultdict(float)
        self.team_region: dict[str, str] = {}
        self.intl: dict[str, list] = defaultdict(list)          # (date, wins, games)
        self.h2h: dict[tuple, list] = defaultdict(list)          # (date, wins_first, games)
        self.player_worlds: dict[str, list] = defaultdict(list)  # player -> dates
        self.year: int | None = None

    # ------------------------------------------------------------------ ratings
    def region_of(self, team: str) -> str:
        return self.team_region.get(team, "LCP")

    def rating(self, team: str) -> float:
        return self.region_rating.get(self.region_of(team), 0.0) + self.team_off[team]

    def new_year(self, year: int) -> None:
        if self.year is not None and year != self.year:
            for t in self.team_off:
                self.team_off[t] *= self.regress
        self.year = year

    def update(self, row, intl: bool) -> None:
        a, b, wa, wb = row.team_a, row.team_b, int(row.wins_a), int(row.wins_b)
        n = wa + wb
        if n == 0:
            return
        if not intl:
            self.team_region[a] = self.team_region[b] = row.region
        e = 1 / (1 + 10 ** ((self.rating(b) - self.rating(a)) / 400))
        surprise = wa - n * e
        self.team_off[a] += self.k_long * surprise
        self.team_off[b] -= self.k_long * surprise
        ra, rb = self.region_of(a), self.region_of(b)
        if intl and ra != rb:
            self.region_rating[ra] = self.region_rating.get(ra, 0) + self.k_region * surprise
            self.region_rating[rb] = self.region_rating.get(rb, 0) - self.k_region * surprise
        if intl:
            self.intl[a].append((row.date, wa, n))
            self.intl[b].append((row.date, wb, n))
        key = tuple(sorted((a, b)))
        self.h2h[key].append((row.date, wa if key[0] == a else wb, n))

    def add_worlds_players(self, date, players: list) -> None:
        for p in players:
            if p:
                self.player_worlds[p].append(date)

    # ------------------------------------------------------------------ features
    def _intl_wr(self, team: str, date) -> float:
        rec = [(w, n) for d, w, n in self.intl[team] if date - d <= TWO_YEARS]
        w = sum(x[0] for x in rec)
        n = sum(x[1] for x in rec)
        return (w + 0.5 * self.intl_prior) / (n + self.intl_prior) - 0.5

    def _h2h(self, a: str, b: str, date) -> float:
        key = tuple(sorted((a, b)))
        rec = [(w, n) for d, w, n in self.h2h[key] if date - d <= TWO_YEARS]
        w = sum(x[0] for x in rec)
        n = sum(x[1] for x in rec)
        share = (w + 0.5 * self.h2h_prior) / (n + self.h2h_prior) - 0.5
        return share if key[0] == a else -share

    def _worlds_exp(self, roster: list, date) -> float:
        if not roster:
            return 0.0
        return float(np.mean([math.log1p(sum(1 for d in self.player_worlds.get(p, [])
                                              if date - d <= THREE_YEARS)) for p in roster]))

    def features(self, a: str, b: str, date, roster_a: list, roster_b: list,
                 cont_a: float = 1.0, cont_b: float = 1.0) -> dict:
        """Team-level records are scaled by roster continuity: a record earned by
        players who have since left says little about the current team."""
        return {
            "elo_long": (self.rating(a) - self.rating(b)) / 400,
            "region": (self.region_rating.get(self.region_of(a), 0)
                       - self.region_rating.get(self.region_of(b), 0)) / 400,
            "intl_wr": cont_a * self._intl_wr(a, date) - cont_b * self._intl_wr(b, date),
            "h2h": min(cont_a, cont_b) * self._h2h(a, b, date),
            "worlds_exp": self._worlds_exp(roster_a, date) - self._worlds_exp(roster_b, date),
        }
