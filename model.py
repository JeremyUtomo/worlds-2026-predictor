"""Game win-probability model: L2 logistic regression on feature differences.

The model is symmetric by construction (no intercept, every game is also added
mirrored), so P(A beats B) = 1 - P(B beats A). Series probabilities come from
the per-game probability analytically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config
from features import GROUPS, feature_names


def p_series(p, bo: int):
    p = np.asarray(p, dtype=float)
    if bo == 1:
        return p
    if bo == 3:
        return p * p * (3 - 2 * p)
    if bo == 5:
        return p ** 3 * (10 - 15 * p + 6 * p * p)
    raise ValueError(bo)


def kind_of(bo: int) -> str:
    return f"bo{int(bo)}"


def apply_temperature(p, T: float):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) / T
    return 1 / (1 + np.exp(-z))


def expand_games(table: pd.DataFrame, cols: list[str], intl_weight: float):
    """One sample per game played, plus the mirrored sample."""
    X = table[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X)
    w_series = np.where(table.league.isin(config.INTL_LEAGUES), intl_weight, 1.0)
    xs, ys, ws = [], [], []
    for x, wa, wb, w in zip(X, table.wins_a, table.wins_b, w_series):
        for y, n in ((1, wa), (0, wb)):
            if n:
                xs += [x, -x]
                ys += [y, 1 - y]
                ws += [w * n, w * n]
    return np.array(xs), np.array(ys), np.array(ws)


class GameModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cols = feature_names(cfg["model"]["groups"])
        self.scale = None
        self.clf = None

    def fit(self, table: pd.DataFrame) -> "GameModel":
        t = table[table.date >= pd.Timestamp(self.cfg["model"]["train_since"], tz="UTC")]
        X, y, w = expand_games(t, self.cols, self.cfg["model"]["intl_weight"])
        self.scale = np.sqrt((X ** 2).mean(axis=0)) + 1e-9
        self.clf = LogisticRegression(C=self.cfg["model"]["l2_C"], fit_intercept=False,
                                      max_iter=2000)
        self.clf.fit(X / self.scale, y, sample_weight=w / w.mean())
        return self

    @property
    def coef(self) -> pd.Series:
        return pd.Series(self.clf.coef_[0] / self.scale, index=self.cols)

    def logit(self, feats: pd.DataFrame | dict) -> np.ndarray:
        df = pd.DataFrame([feats]) if isinstance(feats, dict) else feats
        X = np.nan_to_num(df[self.cols].to_numpy(dtype=float))
        return X @ self.coef.to_numpy()

    def p_game(self, feats) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logit(feats)))

    def contributions(self, feats: dict) -> dict:
        """Logit contribution of each feature group (positive favours team A)."""
        c = self.coef
        out = {}
        for g in self.cfg["model"]["groups"]:
            out[g] = float(sum(c[f] * (feats.get(f) or 0.0) for f in GROUPS[g]))
        return out
