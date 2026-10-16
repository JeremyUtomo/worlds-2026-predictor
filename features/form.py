"""Form group: a fast rating that drifts back to the long-run rating over time,
plus exponentially weighted recent results and early-game gold."""
from __future__ import annotations

from collections import defaultdict


def _half() -> float:   # module-level so the state can be pickled
    return 0.5


class Form:
    def __init__(self, cfg: dict, history):
        self.k_fast = cfg["elo"]["k_fast"]
        self.half_life = cfg["elo"]["fast_half_life_days"]
        n = cfg["form"]["ewma_games"]
        self.alpha = 2 / (n + 1)
        self.alpha_short = 2 / (4 + 1)
        self.h = history
        self.fast_off: dict[str, float] = {}
        self.last_seen: dict[str, object] = {}
        self.win = defaultdict(_half)
        self.win_short = defaultdict(_half)
        self.gd15 = defaultdict(float)
        self.gdm = defaultdict(float)

    def _fast(self, team: str, date) -> float:
        long = self.h.team_off[team]
        off = self.fast_off.get(team, long)
        last = self.last_seen.get(team)
        if last is not None:
            days = (date - last).total_seconds() / 86400
            off = long + (off - long) * 0.5 ** (days / self.half_life)
        return off

    def rating(self, team: str, date) -> float:
        return self.h.region_rating.get(self.h.region_of(team), 0) + self._fast(team, date)

    def update_series(self, row) -> None:
        a, b, wa, wb = row.team_a, row.team_b, int(row.wins_a), int(row.wins_b)
        n = wa + wb
        if n == 0:
            return
        fa, fb = self._fast(a, row.date), self._fast(b, row.date)
        e = 1 / (1 + 10 ** ((self.rating(b, row.date) - self.rating(a, row.date)) / 400))
        s = wa - n * e
        self.fast_off[a], self.fast_off[b] = fa + self.k_fast * s, fb - self.k_fast * s
        self.last_seen[a] = self.last_seen[b] = row.date
        for t, w in ((a, wa), (b, wb)):
            for i in range(n):
                y = 1.0 if i < w else 0.0
                self.win[t] += self.alpha * (y - self.win[t])
                self.win_short[t] += self.alpha_short * (y - self.win_short[t])

    def update_game(self, team: str, gd15: float | None, gdm: float | None) -> None:
        if gd15 is not None and gd15 == gd15:
            self.gd15[team] += self.alpha * (gd15 - self.gd15[team])
        if gdm is not None and gdm == gdm:
            self.gdm[team] += self.alpha * (gdm - self.gdm[team])

    def features(self, a: str, b: str, date) -> dict:
        return {
            "elo_fast": (self.rating(a, date) - self.rating(b, date)) / 400,
            "recent_wr": self.win[a] - self.win[b],
            "trend": (self.win_short[a] - self.win[a]) - (self.win_short[b] - self.win[b]),
            "gd15": (self.gd15[a] - self.gd15[b]) / 1000,
            "gdm": (self.gdm[a] - self.gdm[b]) / 100,
        }
