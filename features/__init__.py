"""Walk-forward feature engine.

``Engine.run()`` walks every series in date order. For each one it records the
features *as they stood before the match*, then feeds the result in. After the
run the engine holds today's state, so ``Engine.features(a, b, now)`` gives
pre-match features for any pairing (that is what the simulator uses).
"""
from __future__ import annotations

import pandas as pd

import config
from .form import Form
from .history import History
from .meta import Meta, picks_table
from .meta_fit import MetaFit
from .player_form import PlayerForm
from .roster import Roster

GROUPS = {
    "form": ["elo_fast", "recent_wr", "trend", "gd15", "gdm"],
    "history": ["elo_long", "region", "intl_wr", "h2h", "worlds_exp"],
    "roster": ["roster_gap", "continuity"],
    "roster_full": ["elo_players", "elo_roster", "continuity", "roster_gap"],
    "player_form": ["pf_lane15", "pf_cs15", "pf_kda"],
    "meta_fit": ["champ_fit", "meta_mastery", "depth", "ban_vuln", "style_fit"],
}
ALL_FEATURES = list(dict.fromkeys(f for fs in GROUPS.values() for f in fs))


def feature_names(groups: list[str]) -> list[str]:
    return [f for g in groups for f in GROUPS[g]]


class Engine:
    def __init__(self, cfg: dict, series: pd.DataFrame, games: pd.DataFrame):
        self.cfg = cfg
        self.series = series
        self.games = games
        self.picks = picks_table(games) if len(games) else pd.DataFrame(
            columns=["date", "game_id", "league", "patch", "team", "player", "role", "champ", "win"])
        self.history = History(cfg)
        self.form = Form(cfg, self.history)
        self.meta = Meta(cfg, self.picks, games)
        self.fit = MetaFit(cfg, self.meta, self.picks)
        self.rost = Roster(cfg, self.history)
        self.pform = PlayerForm(cfg)
        self.roster: dict[str, list] = {}
        self.by_match = {m: list(g.sort_values("number").itertuples())
                         for m, g in games.groupby("match_id")} if len(games) else {}
        self.now = None

    def __getstate__(self):
        # per-match game rows are only needed while walking history (and are not picklable)
        state = self.__dict__.copy()
        state["by_match"] = {}
        return state

    def lineup(self, team: str, match_id: str | None = None) -> list:
        """Starting five: the one announced for this match if known, else the last one seen."""
        for g in self.by_match.get(match_id, [])[:1]:
            for side in ("blue", "red"):
                if getattr(g, side) == team:
                    players = [p for p in getattr(g, f"{side}_players") if p]
                    if len(players) == 5:
                        return players
        return self.roster.get(team, [])

    def features(self, a: str, b: str, date, match_id: str | None = None) -> dict:
        ra, rb = self.lineup(a, match_id), self.lineup(b, match_id)
        f = {}
        rf, ca, cb = self.rost.features(a, b, ra, rb, date)
        f.update(self.form.features(a, b, date))
        if self.cfg["roster"].get("blend", True):
            # judge each team on the players actually starting: the more the roster
            # changed, the more the rating comes from those players' own records
            ba = ca * self.form.rating(a, date) + (1 - ca) * self.rost.player_rating(a, ra)
            bb = cb * self.form.rating(b, date) + (1 - cb) * self.rost.player_rating(b, rb)
            f["elo_fast"] = (ba - bb) / 400
        f.update(self.history.features(a, b, date, ra, rb, ca, cb))
        f.update(rf)
        f.update(self.pform.features(ra, rb))
        f.update(self.fit.features(a, b, ra, rb, date, self.form.gd15[a], self.form.gd15[b]))
        return f

    def _ingest(self, row) -> None:
        intl = row.league in config.INTL_LEAGUES
        self.form.update_series(row)
        self.history.update(row, intl)
        for g in self.by_match.get(row.match_id, []):
            self.pform.update_game(g)
            if g.blue and g.red:
                self.rost.update_game(row.date, g.blue, g.red, [p for p in g.blue_players if p],
                                      [p for p in g.red_players if p], g.winner == "blue")
            for side, other in (("blue", "red"), ("red", "blue")):
                team = getattr(g, side)
                if team is None:
                    continue
                sign = 1 if side == "blue" else -1
                gd15 = g.gd15 * sign if g.gd15 == g.gd15 else None
                end_b, end_r = g.blue_end, g.red_end
                gdm = sign * (end_b["gold"] - end_r["gold"]) / max(g.minutes, 15)
                self.form.update_game(team, gd15, gdm)
                won = g.winner == side
                self.fit.update_game(team, gd15 is not None and gd15 < -1000, won)
                players = [p for p in getattr(g, f"{side}_players") if p]
                if len(players) == 5:
                    self.roster[team] = players
                if row.league == "worlds":
                    self.history.add_worlds_players(row.date, players)

    def run(self, until=None) -> pd.DataFrame:
        s = self.series
        s = s[(s.state == "completed") & s.wins_a.notna() & s.wins_b.notna()
              & (s.team_a != "TBD") & (s.team_b != "TBD")]
        if until is not None:
            s = s[s.date < until]
        rows = []
        for r in s.sort_values("date").itertuples():
            if self.history.year is not None and r.date.year != self.history.year:
                self.rost.new_year(self.history.regress)
            self.history.new_year(r.date.year)
            f = self.features(r.team_a, r.team_b, r.date, r.match_id)
            rows.append({"match_id": r.match_id, "date": r.date, "league": r.league,
                         "block": r.block, "bo": int(r.bo), "team_a": r.team_a,
                         "team_b": r.team_b, "wins_a": int(r.wins_a), "wins_b": int(r.wins_b),
                         **f})
            self._ingest(r)
        self.now = s.date.max() if len(s) else None
        return pd.DataFrame(rows)
