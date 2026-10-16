"""Monte Carlo of the rest of Worlds 2026: Play-In -> Swiss -> knockouts.

Real results are locked in from the lolesports schedule, so re-running after
each match day gives updated odds. Swiss rules (as Riot runs them):
* 16 teams, play until 3 wins or 3 losses, paired within the same record;
* no rematches; round 1 avoids same-region pairs;
* Bo1 unless a team can advance or be eliminated in that match (then Bo3).
Knockouts are Bo5 single elimination; the draw comes from settings/bracket.yaml
once Riot publishes it, otherwise 3-0 teams are drawn against 3-2 teams.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yaml

import config


class Probs:
    """Series win probabilities between Worlds teams, by format."""

    def __init__(self, teams: list[str], p: dict[int, np.ndarray]):
        self.idx = {t: i for i, t in enumerate(teams)}
        self.p = p

    def __call__(self, a: str, b: str, bo: int) -> float:
        return float(self.p[bo][self.idx[a], self.idx[b]])


def _results(series: pd.DataFrame) -> list[tuple[frozenset, str]]:
    """Completed series in date order as (pair, winner)."""
    done = series[series.state == "completed"].sort_values("date")
    return [(frozenset((r.team_a, r.team_b)), r.team_a if r.wins_a > r.wins_b else r.team_b)
            for r in done.itertuples()]


class _Known:
    """Hands out each real result once, so a rematch later in a bracket is simulated."""

    def __init__(self, results):
        self.results = results
        self.used = set()

    def __call__(self, a: str, b: str):
        pair = frozenset((a, b))
        for i, (p, w) in enumerate(self.results):
            if p == pair and i not in self.used:
                self.used.add(i)
                return w
        return None


def _play(probs: Probs, a: str, b: str, bo: int, rng: random.Random) -> str:
    return a if rng.random() < probs(a, b, bo) else b


# --------------------------------------------------------------------------- play-in
def play_in(results: list, probs: Probs, rng: random.Random) -> str:
    """Double elimination, 4 teams, all Bo5. Returns the team that reaches Swiss."""
    known = _Known(results)

    def match(a, b):
        return known(a, b) or _play(probs, a, b, 5, rng)

    a1, b1 = "KC", "C9"
    a2, b2 = "MVK", "FUR"
    w1 = match(a1, b1)
    l1 = b1 if w1 == a1 else a1
    w2 = match(a2, b2)
    l2 = b2 if w2 == a2 else a2
    wu = match(w1, w2)
    lu = w2 if wu == w1 else w1
    wl = match(l1, l2)
    wlf = match(lu, wl)
    return match(wu, wlf)


# --------------------------------------------------------------------------- swiss
def _pair(group: list[str], played: set, region: dict, avoid_region: bool,
          rng: random.Random) -> list[tuple]:
    best = None
    for attempt in range(60):
        g = group[:]
        rng.shuffle(g)
        pairs = list(zip(g[::2], g[1::2]))
        rematch = sum(frozenset(p) in played for p in pairs)
        same = sum(region[a] == region[b] for a, b in pairs) if avoid_region else 0
        cost = rematch * 10 + same
        if best is None or cost < best[0]:
            best = (cost, pairs)
        if cost == 0:
            break
    return best[1]


def swiss_state(worlds: pd.DataFrame):
    sw = worlds[worlds.block.str.contains("Swiss", case=False)].sort_values("date")
    rec = defaultdict(lambda: [0, 0])
    played: set = set()
    fixed = []
    for r in sw.itertuples():
        if "TBD" in (r.team_a, r.team_b):
            continue
        if r.state == "completed":
            w, l = (r.team_a, r.team_b) if r.wins_a > r.wins_b else (r.team_b, r.team_a)
            rec[w][0] += 1
            rec[l][1] += 1
            played.add(frozenset((r.team_a, r.team_b)))
        else:
            fixed.append((r.team_a, r.team_b, int(r.bo)))
    return rec, played, fixed


def swiss(teams: list[str], probs: Probs, rng: random.Random, state=None) -> dict:
    region = config.WORLDS_2026
    rec0, played0, fixed = state or ({}, set(), [])
    rec = {t: list(rec0.get(t, (0, 0))) for t in teams}
    played = set(played0)
    fixed = [f for f in fixed if f[0] in rec and f[1] in rec]
    while True:
        active = [t for t in teams if rec[t][0] < 3 and rec[t][1] < 3]
        if not active:
            break
        rnd = min(sum(rec[t]) for t in active)
        todo = [t for t in active if sum(rec[t]) == rnd]
        pairs = []
        for a, b, bo in fixed:
            if a in todo and b in todo:
                pairs.append((a, b, bo))
                todo.remove(a)
                todo.remove(b)
        fixed = [f for f in fixed if (f[0], f[1], f[2]) not in pairs]
        groups = defaultdict(list)
        for t in todo:
            groups[tuple(rec[t])].append(t)
        for g in groups.values():
            for a, b in _pair(g, played, region, rnd == 0, rng):
                stake = max(rec[a][0], rec[a][1]) == 2
                pairs.append((a, b, 3 if stake else 1))
        for a, b, bo in pairs:
            w = _play(probs, a, b, bo, rng)
            l = b if w == a else a
            rec[w][0] += 1
            rec[l][1] += 1
            played.add(frozenset((a, b)))
    return {t: f"{w}-{l}" for t, (w, l) in rec.items()}


# --------------------------------------------------------------------------- knockouts
def load_bracket() -> list[str] | None:
    """Quarterfinal order from settings/bracket.yaml: [QF1a, QF1b, QF2a, QF2b, ...];
    QF1/QF2 meet in SF1, QF3/QF4 in SF2."""
    if not config.BRACKET_FILE.exists():
        return None
    b = yaml.safe_load(config.BRACKET_FILE.read_text()) or {}
    qf = b.get("quarterfinals")
    return [t for pair in qf for t in pair] if qf else None


def draw_bracket(records: dict, rng: random.Random) -> list[str]:
    adv = [t for t, r in records.items() if r.startswith("3")]
    s30 = [t for t in adv if records[t] == "3-0"]
    s32 = [t for t in adv if records[t] == "3-2"]
    rest = [t for t in adv if t not in s30]
    rng.shuffle(s32)
    rng.shuffle(rest)
    qfs = []
    for t in s30:
        opp = s32.pop() if s32 else rest.pop()
        if opp in rest:
            rest.remove(opp)
        qfs.append((t, opp))
    rest = [t for t in rest if all(t not in q for q in qfs)]
    while rest:
        qfs.append((rest.pop(), rest.pop()))
    # the two 3-0 teams go to opposite halves
    if len(s30) == 2:
        order = [qfs[0], qfs[2], qfs[1], qfs[3]]
    else:
        order = qfs
    return [t for q in order for t in q]


def knockouts(order: list[str], probs: Probs, rng: random.Random, results: list) -> list[list[str]]:
    known = _Known(results)
    rounds = [order]
    cur = order
    while len(cur) > 1:
        nxt = []
        for a, b in zip(cur[::2], cur[1::2]):
            nxt.append(known(a, b) or _play(probs, a, b, 5, rng))
        rounds.append(nxt)
        cur = nxt
    return rounds


# --------------------------------------------------------------------------- driver
SWISS_DIRECT = [t for t in config.WORLDS_2026 if t not in config.PLAY_IN]


def run(worlds: pd.DataFrame, probs: Probs, n: int = 20000, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    state = swiss_state(worlds)
    sw = worlds[worlds.block.str.contains("Swiss", case=False)]
    known = set(sw.team_a) | set(sw.team_b)
    pi_known = [t for t in config.PLAY_IN if t in known]
    ko = _results(worlds[worlds.block.str.contains("final", case=False)])
    pi_res = _results(worlds[worlds.block.str.contains("Play", case=False)])
    bracket = load_bracket()
    counts = defaultdict(Counter)
    for _ in range(n):
        piw = pi_known[0] if pi_known else play_in(pi_res, probs, rng)
        counts[piw]["play_in_win"] += 1
        teams = SWISS_DIRECT + [piw]
        recs = swiss(teams, probs, rng, state)
        for t, r in recs.items():
            counts[t][r] += 1
            if r.startswith("3"):
                counts[t]["adv"] += 1
        order = bracket or draw_bracket(recs, rng)
        rounds = knockouts(order, probs, rng, ko)
        for stage, teams_in in zip(["qf", "sf", "final", "champ"], rounds):
            for t in teams_in:
                counts[t][stage] += 1
    cols = ["play_in_win", "3-0", "3-1", "3-2", "2-3", "1-3", "0-3", "adv", "sf", "final", "champ"]
    df = pd.DataFrame({t: {c: counts[t][c] / n for c in cols} for t in config.WORLDS_2026}).T
    return df.sort_values(["champ", "adv"], ascending=False)
