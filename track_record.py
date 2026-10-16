"""Prediction log + scoring after the match + mistakes report.

* ``log_predictions`` writes a row for every Worlds match whose two teams are
  known **and that has not started yet**. Re-running before the match
  overwrites that row with the latest view; once a match starts its row is frozen.
* ``score`` joins the log with results: favourite won?, log loss, Brier.
* ``mistakes`` lists upsets and which feature groups pushed the wrong way.
* ``evidence`` = backtest + scored live predictions, used by calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import data
from model import kind_of

LOG_COLS = ["match_id", "saved_utc", "date", "block", "bo", "team_a", "team_b",
            "p_game", "p_a", "c_form", "c_history", "c_roster", "c_meta_fit"]
GROUP_NAMES = ("form", "history", "roster", "meta_fit")


def read_log() -> pd.DataFrame:
    if not config.PREDICTION_LOG.exists():
        return pd.DataFrame(columns=LOG_COLS)
    df = pd.read_csv(config.PREDICTION_LOG, dtype={"match_id": str})
    df["date"] = pd.to_datetime(df.date, utc=True)
    return df


def log_predictions(built) -> pd.DataFrame:
    now = data.now()
    w = data.worlds_matches(built.series)
    upcoming = w[(w.state == "unstarted") & (w.team_a != "TBD") & (w.team_b != "TBD")
                 & (w.date > now)]
    log = read_log()
    rows = []
    for r in upcoming.itertuples():
        ex = built.explain(r.team_a, r.team_b)
        rows.append({"match_id": r.match_id, "saved_utc": now.isoformat(timespec="seconds"),
                     "date": r.date, "block": r.block, "bo": int(r.bo),
                     "team_a": r.team_a, "team_b": r.team_b,
                     "p_game": round(built.p_game(r.team_a, r.team_b), 4),
                     "p_a": round(built.p_match(r.team_a, r.team_b, int(r.bo)), 4),
                     **{f"c_{g}": round(ex["groups"].get(g, 0.0), 4)
                        for g in GROUP_NAMES}})
    if rows:
        new = pd.DataFrame(rows)
        log = pd.concat([log[~log.match_id.isin(new.match_id)], new], ignore_index=True)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        log.sort_values("date").to_csv(config.PREDICTION_LOG, index=False)
    return pd.DataFrame(rows)


def log_forecast(sim: pd.DataFrame, stage: str) -> None:
    snap = sim.reset_index().rename(columns={"index": "team"})
    snap.insert(0, "stage", stage)
    snap.insert(0, "saved_utc", data.now().isoformat(timespec="seconds"))
    header = not config.FORECAST_LOG.exists()
    snap.to_csv(config.FORECAST_LOG, mode="a", header=header, index=False)


def score(series: pd.DataFrame) -> pd.DataFrame:
    log = read_log()
    if log.empty:
        return log
    res = series[series.state == "completed"][["match_id", "team_a", "wins_a", "wins_b"]]
    res = res.rename(columns={"team_a": "res_a"})
    df = log.merge(res, on="match_id", how="left")
    done = df.wins_a.notna()
    # result from the logged team_a's point of view
    a_won = np.where(df.res_a == df.team_a, df.wins_a > df.wins_b, df.wins_b > df.wins_a)
    df["status"] = np.where(done, "Scored", "Awaiting result")
    df["y"] = np.where(done, a_won.astype(float), np.nan)
    p = df.p_a.clip(1e-4, 1 - 1e-4)
    df["fav"] = np.where(df.p_a >= 0.5, df.team_a, df.team_b)
    df["fav_won"] = np.where(done, (df.p_a >= 0.5) == (df.y == 1), np.nan)
    df["logloss"] = np.where(done, -(df.y * np.log(p) + (1 - df.y) * np.log(1 - p)), np.nan)
    df["brier"] = np.where(done, (df.p_a - df.y) ** 2, np.nan)
    return df


def summary(scored: pd.DataFrame) -> dict:
    if scored.empty or "status" not in scored:
        return {"scored": 0}
    s = scored[scored.status == "Scored"]
    if s.empty:
        return {"scored": 0}
    return {"scored": len(s), "favourite_won": float(s.fav_won.mean()),
            "logloss": float(s.logloss.mean()), "brier": float(s.brier.mean()),
            "coin_flip_logloss": float(np.log(2))}


def mistakes(scored: pd.DataFrame) -> pd.DataFrame:
    """Upsets, and which feature groups backed the favourite that lost."""
    if scored.empty or "status" not in scored:
        return scored
    s = scored[(scored.status == "Scored") & (scored.fav_won == 0)].copy()
    if s.empty:
        return s
    sign = np.where(s.p_a >= 0.5, 1, -1)
    out = s[["date", "block", "team_a", "team_b", "fav", "p_a"]].copy()
    out["fav_prob"] = np.where(s.p_a >= 0.5, s.p_a, 1 - s.p_a)
    for g in GROUP_NAMES:
        if f"c_{g}" in s:
            out[f"misled_by_{g}"] = (s[f"c_{g}"].fillna(0) * sign).round(3)   # >0: pushed toward the loser
    groups = out[[c for c in out.columns if c.startswith("misled_by_")]]
    out["main_culprit"] = groups.idxmax(axis=1).str.replace("misled_by_", "")
    return out.sort_values("fav_prob", ascending=False)


def evidence(backtest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calibration evidence: walk-forward backtest + scored live predictions."""
    parts = []
    if backtest is not None and not backtest.empty:
        parts.append(backtest[["kind", "p", "y"]])
    if config.PREDICTION_LOG.exists():
        sc = score(data.load_series(refresh=False))
        sc = sc[sc.status == "Scored"] if "status" in sc else sc.iloc[0:0]
        if not sc.empty:
            parts.append(pd.DataFrame({"kind": [kind_of(b) for b in sc.bo],
                                       "p": sc.p_a, "y": sc.y}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["kind", "p", "y"])
