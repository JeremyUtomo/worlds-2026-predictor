"""Player form: how each player has been playing lately, wherever they played it.

Per player, exponentially weighted over their recent games:
* lane15 - gold lead at 15 minutes over the opposing player in the same role
* cs15   - CS lead at 15 minutes over the same opponent
* kda    - log((kills + assists + 1) / (deaths + 1)) at the end of the game

A team's player form is the average over the five who start, so it moves with
transfers and substitutions instead of sticking to the team name.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

STATS = ("lane15", "cs15", "kda")
ROLES = ("top", "jungle", "mid", "bottom", "support")


def _zero_stats() -> dict:
    return {s: 0.0 for s in STATS}


class PlayerForm:
    def __init__(self, cfg: dict):
        n = cfg.get("player_form", {}).get("ewma_games", 10)
        self.alpha = 2 / (n + 1)
        self.form: dict[str, dict] = defaultdict(_zero_stats)
        self.games: dict[str, int] = defaultdict(int)
        self.recent: dict[str, list] = defaultdict(list)   # player -> last results for display
        self.name: dict[str, str] = {}
        self.role: dict[str, str] = {}

    def update_game(self, g) -> None:
        p15b, p15r = getattr(g, "blue_p15", None), getattr(g, "red_p15", None)
        peb, per = getattr(g, "blue_pend", None), getattr(g, "red_pend", None)
        if not isinstance(peb, list) or not isinstance(per, list):
            return
        has15 = isinstance(p15b, list) and isinstance(p15r, list)
        for side, opp in (("blue", "red"), ("red", "blue")):
            players = getattr(g, f"{side}_players")
            names = getattr(g, f"{side}_names")
            champs = getattr(g, f"{side}_champs")
            mine_end = peb if side == "blue" else per
            mine15 = (p15b if side == "blue" else p15r) if has15 else None
            theirs15 = (p15r if side == "blue" else p15b) if has15 else None
            won = g.winner == side
            for i, pid in enumerate(players):
                if not pid or i >= 5:
                    continue
                gold, cs, k, d, a = mine_end[i]
                obs = {"kda": math.log((k + a + 1) / (d + 1))}
                if mine15 is not None:
                    obs["lane15"] = (mine15[i][0] - theirs15[i][0]) / 500
                    obs["cs15"] = (mine15[i][1] - theirs15[i][1]) / 10
                f = self.form[pid]
                if self.games[pid] == 0:
                    f["kda"] = 1.0
                for s, v in obs.items():
                    f[s] += self.alpha * (v - f[s])
                self.games[pid] += 1
                self.name[pid] = names[i] if i < len(names) else pid
                self.role[pid] = ROLES[i]
                rec = self.recent[pid]
                rec.append((g.date, champs[i], won, k, d, a, obs.get("lane15")))
                if len(rec) > 10:
                    del rec[0]

    def team(self, roster: list) -> dict:
        if not roster:
            return {f"pf_{s}": 0.0 for s in STATS}
        vals = np.array([[self.form[p][s] if self.games.get(p) else (1.0 if s == "kda" else 0.0)
                          for s in STATS] for p in roster])
        return {f"pf_{s}": float(v) for s, v in zip(STATS, vals.mean(axis=0))}

    def features(self, ra: list, rb: list) -> dict:
        fa, fb = self.team(ra), self.team(rb)
        return {k: fa[k] - fb[k] for k in fa}

    def card(self, roster: list) -> list[dict]:
        """Per-player summary for the dashboard."""
        out = []
        for p in roster:
            f = self.form[p] if self.games.get(p) else {s: np.nan for s in STATS}
            rec = self.recent.get(p, [])
            out.append({"player": self.name.get(p, p), "role": self.role.get(p, ""),
                        "lane gold @15": round(f["lane15"] * 500) if f["lane15"] == f["lane15"] else None,
                        "CS @15": round(f["cs15"] * 10, 1) if f["cs15"] == f["cs15"] else None,
                        "KDA form": round(math.exp(f["kda"]), 2) if f["kda"] == f["kda"] else None,
                        "last 5": "".join("W" if r[2] else "L" for r in rec[-5:]),
                        "recent champs": ", ".join(dict.fromkeys(r[1] for r in reversed(rec)))[:60],
                        "games": self.games.get(p, 0)})
        return out
