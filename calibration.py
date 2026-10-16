"""Self-calibration: learn how much to trust the model, per series format.

Same scheme as fastf1-predictor: re-predict past international events
walk-forward, then fit one temperature per kind (bo1 / bo3 / bo5) that rescales
the series logit:  p' = sigmoid(logit(p) / T).  T < 1 sharpens, T > 1 flattens.
With few matches the learned value is shrunk toward 1:
    T_used = 1 + (T_raw - 1) * n / (n + SHRINK)
Logged live Worlds predictions are added to the evidence as they are scored, so
every new result moves the temperatures.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from model import apply_temperature

GRID = np.round(np.arange(0.5, 3.0001, 0.05), 2)


def _nll(p, y) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperatures(evidence: pd.DataFrame, cfg: dict) -> dict:
    """evidence: columns kind, p (series prob for team A), y (1 if A won)."""
    min_n, shrink = cfg["calibration"]["min_events"], cfg["calibration"]["shrink"]
    out = {}
    for kind in ("bo1", "bo3", "bo5"):
        e = evidence[evidence.kind == kind]
        n = len(e)
        if n < min_n:
            out[kind] = {"T": 1.0, "raw": None, "n": n, "learned": False}
            continue
        losses = [_nll(apply_temperature(e.p, T), e.y) for T in GRID]
        raw = float(GRID[int(np.argmin(losses))])
        used = 1 + (raw - 1) * n / (n + shrink)
        out[kind] = {"T": round(used, 3), "raw": raw, "n": n, "learned": True,
                     "logloss_default": round(_nll(e.p, e.y), 4),
                     "logloss_learned": round(_nll(apply_temperature(e.p, used), e.y), 4)}
    return out


def save(temps: dict) -> None:
    config.CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CALIBRATION_FILE.write_text(json.dumps(temps, indent=2))


def load() -> dict:
    if config.CALIBRATION_FILE.exists():
        return json.loads(config.CALIBRATION_FILE.read_text())
    return {k: {"T": 1.0} for k in ("bo1", "bo3", "bo5")}


def calibration_table(evidence: pd.DataFrame, temps: dict, bins: int = 5) -> pd.DataFrame:
    """Predicted vs observed win rate per probability bin (favourite's view)."""
    e = evidence.copy()
    e["p_cal"] = [apply_temperature(p, temps.get(k, {}).get("T", 1.0))
                  for p, k in zip(e.p, e.kind)]
    fav = np.where(e.p_cal >= 0.5, e.p_cal, 1 - e.p_cal)
    hit = np.where(e.p_cal >= 0.5, e.y, 1 - e.y)
    e = pd.DataFrame({"pred": fav, "hit": hit})
    e["bin"] = pd.cut(e.pred, np.linspace(0.5, 1.0, bins + 1), include_lowest=True)
    return e.groupby("bin", observed=True).agg(predicted=("pred", "mean"),
                                               observed=("hit", "mean"), n=("hit", "size"))
