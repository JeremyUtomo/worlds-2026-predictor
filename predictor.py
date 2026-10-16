"""Glue: data -> features -> model -> calibrated series probabilities for Worlds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import calibration
import config
import data
from features import Engine
from model import GameModel, apply_temperature, kind_of, p_series
from simulate import Probs


@dataclass
class Built:
    cfg: dict
    series: pd.DataFrame
    games: pd.DataFrame
    engine: Engine
    table: pd.DataFrame
    model: GameModel
    temps: dict
    now: pd.Timestamp

    def p_game(self, a: str, b: str) -> float:
        return float(self.model.p_game(self.engine.features(a, b, self.now))[0])

    def p_match(self, a: str, b: str, bo: int) -> float:
        p = p_series(self.p_game(a, b), bo)
        return float(apply_temperature(p, self.temps.get(kind_of(bo), {}).get("T", 1.0)))

    def explain(self, a: str, b: str) -> dict:
        f = self.engine.features(a, b, self.now)
        return {"features": f, "groups": self.model.contributions(f)}

    def probs(self, teams: list[str] | None = None) -> Probs:
        teams = teams or list(config.WORLDS_2026)
        g = np.full((len(teams), len(teams)), 0.5)
        for i, a in enumerate(teams):
            for j, b in enumerate(teams):
                if i < j:
                    g[i, j] = self.p_game(a, b)
                    g[j, i] = 1 - g[i, j]
        mats = {}
        for bo in (1, 3, 5):
            T = self.temps.get(kind_of(bo), {}).get("T", 1.0)
            m = apply_temperature(p_series(g, bo), T)
            np.fill_diagonal(m, 0.5)
            mats[bo] = m
        return Probs(teams, mats)


def build(refresh: bool = True, cfg: dict | None = None, fetch_games: bool = True) -> Built:
    cfg = cfg or config.load_config()
    series = data.load_series(refresh=refresh)
    games = data.load_games(series, refresh=refresh and fetch_games, progress=False) \
        if fetch_games or (config.DATA_DIR / "games.pkl").exists() else pd.DataFrame()
    engine = Engine(cfg, series, games)
    table = engine.run()
    model = GameModel(cfg).fit(table)
    now = max(data.now(), table.date.max())
    return Built(cfg, series, games, engine, table, model, calibration.load(), now)
