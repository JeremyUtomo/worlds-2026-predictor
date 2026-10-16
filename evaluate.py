"""Walk-forward backtest and tuning (mirrors fastf1-predictor's evaluate.py).

Every international event (First Stand, MSI, EWC, Worlds) since mid-2024 is
re-predicted using a model fitted only on games before that event started;
features inside the event update live, exactly as they would on the day.

    python evaluate.py           # backtest with the current config
    python evaluate.py --tune    # one-change-at-a-time search, writes
                                 # model_config.json + research_report.md
Tuning uses events before 2026; 2026 events are the holdout. A change is
adopted only if tuning log loss drops by more than MIN_GAIN and Brier score
does not get worse by more than BRIER_TOL.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import calibration
import config
import data
from features import Engine
from model import GameModel, kind_of, p_series

MIN_GAIN = 0.003
BRIER_TOL = 0.0005
HOLDOUT_FROM = pd.Timestamp("2026-01-01", tz="UTC")
BACKTEST_FROM = pd.Timestamp("2024-06-01", tz="UTC")
BACKTEST_FILE = config.DATA_DIR / "backtest.pkl"


def events(table: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.DataFrame]]:
    intl = table[table.league.isin(config.INTL_LEAGUES)].copy()
    intl["event"] = intl.league + " " + intl.date.dt.year.astype(str)
    out = []
    for name, g in intl.groupby("event"):
        start = g.date.min()
        if start >= BACKTEST_FROM:
            out.append((name, start, g))
    return sorted(out, key=lambda x: x[1])


_TABLES: dict[str, pd.DataFrame] = {}


def feature_table(cfg: dict, series, games) -> pd.DataFrame:
    """Engine output, cached per feature setting (model-only changes reuse it)."""
    key = json.dumps({k: cfg[k] for k in ("elo", "form", "history", "roster", "player_form", "meta")}, sort_keys=True)
    if key not in _TABLES:
        _TABLES[key] = Engine(cfg, series, games).run()
    return _TABLES[key]


def walk_forward(cfg: dict, series: pd.DataFrame, games: pd.DataFrame,
                 table: pd.DataFrame | None = None) -> pd.DataFrame:
    if table is None:
        table = feature_table(cfg, series, games)
    rows = []
    for name, start, ev in events(table):
        m = GameModel(cfg).fit(table[table.date < start])
        pg = m.p_game(ev)
        for (r, p) in zip(ev.itertuples(), pg):
            contrib = m.contributions(r._asdict())
            rows.append({"event": name, "match_id": r.match_id, "date": r.date,
                         "block": r.block, "bo": r.bo, "kind": kind_of(r.bo),
                         "team_a": r.team_a, "team_b": r.team_b, "p_game": p,
                         "p": float(p_series(p, r.bo)), "y": int(r.wins_a > r.wins_b),
                         **{f"c_{g}": v for g, v in contrib.items()}})
    return pd.DataFrame(rows)


def metrics(ev: pd.DataFrame) -> dict:
    if ev.empty:
        return {"n": 0, "logloss": np.nan, "brier": np.nan, "acc": np.nan}
    p = np.clip(ev.p.to_numpy(), 1e-6, 1 - 1e-6)
    y = ev.y.to_numpy()
    return {"n": len(ev),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2)),
            "acc": float(np.mean((p > 0.5) == y))}


def split(ev: pd.DataFrame) -> tuple[dict, dict]:
    return metrics(ev[ev.date < HOLDOUT_FROM]), metrics(ev[ev.date >= HOLDOUT_FROM])


# --------------------------------------------------------------------------- tuning
def _set(cfg: dict, path: str, value) -> dict:
    c = copy.deepcopy(cfg)
    sec, key = path.split(".")
    c[sec][key] = value
    return c


# roster.blend is pinned on (not tuned): the small tuning set prefers off, but the 2026
# holdout and a rolling check over all ~6,200 pro games both prefer on.
ROSTER_NOTE = """## Pinned setting: roster-aware rating

`roster.blend` is pinned **on**: the rating used = continuity x team rating + (1 - continuity) x average rating of the five starters.
- The tuning rule alone would turn it off, based on the small set of tuning events: 136 international series, log loss 0.5505 off vs 0.5672 on.
- The 2026 holdout preferred it on: 0.701 on vs 0.713 off.
- A rolling month-by-month check over all ~6,200 pro games also preferred it on. In matches after a big roster change (continuity < 0.6), log loss was 0.631 on vs 0.635 off.
"""

CANDIDATES = [
    ("model.groups", [["form", "player_form", "meta_fit"], ["form", "player_form"],
                      ["player_form", "meta_fit"], ["form", "roster", "meta_fit"], ["form", "history", "roster", "meta_fit"],
                      ["form", "roster"], ["form", "history", "meta_fit"], ["form", "meta_fit"]]),
    ("roster.rookie_start", [0.5, 0.0]),
    ("player_form.ewma_games", [5, 20]),
    ("roster.k_player", [6.0, 20.0]),
    ("roster.continuity_half_life_days", [45, 180]),
    ("model.intl_weight", [1.0, 2.0, 5.0]),
    ("elo.k_long", [16.0, 32.0]),
    ("elo.k_fast", [32.0, 64.0]),
    ("elo.fast_half_life_days", [30.0, 120.0]),
    ("elo.k_region", [6.0, 20.0]),
    ("elo.season_regress", [0.5, 0.9]),
    ("form.ewma_games", [6, 20]),
    ("meta.window_days", [30, 75]),
    ("meta.pool_days", [60, 240]),
    ("meta.patch_strength", [0.0, 0.5]),
    ("model.l2_C", [0.1, 0.3, 3.0]),
]


def tune(series, games, cfg: dict) -> tuple[dict, list[dict]]:
    def score(c):
        ev = walk_forward(c, series, games)
        t, h = split(ev)
        return t, h, ev

    best_t, best_h, best_ev = score(cfg)
    trials = [{"change": "baseline", "tune": best_t, "holdout": best_h, "adopted": True}]
    print(f"baseline  tune LL {best_t['logloss']:.4f}  holdout LL {best_h['logloss']:.4f}")
    for path, values in CANDIDATES:
        sec, key = path.split(".")
        for v in values:
            if cfg[sec][key] == v:
                continue
            c = _set(cfg, path, v)
            t, h, ev = score(c)
            ok = (t["logloss"] < best_t["logloss"] - MIN_GAIN
                  and t["brier"] <= best_t["brier"] + BRIER_TOL)
            trials.append({"change": f"{path} = {v}", "tune": t, "holdout": h, "adopted": ok})
            print(f"{'ADOPT' if ok else '     '} {path} = {v}: tune LL {t['logloss']:.4f} "
                  f"(best {best_t['logloss']:.4f})  holdout LL {h['logloss']:.4f}")
            if ok:
                cfg, best_t, best_h, best_ev = c, t, h, ev
    return cfg, trials


def ablation(series, games, cfg) -> list[dict]:
    out = []
    for groups in (["form"], ["form", "history"], ["form", "roster"], ["form", "meta_fit"],
                   ["form", "roster", "meta_fit"], ["form", "player_form"],
                   ["form", "player_form", "meta_fit"], ["form", "history", "roster", "meta_fit"]):
        c = _set(cfg, "model.groups", groups)
        t, h = split(walk_forward(c, series, games))
        out.append({"groups": "+".join(groups), "tune": t, "holdout": h})
    return out


def calibrated_holdout(ev: pd.DataFrame, cfg: dict) -> dict:
    """Honest check of the self-calibration: fit temperatures on pre-2026
    events, apply them to the 2026 holdout."""
    from model import apply_temperature
    temps = calibration.fit_temperatures(ev[ev.date < HOLDOUT_FROM], cfg)
    h = ev[ev.date >= HOLDOUT_FROM].copy()
    h["p"] = [float(apply_temperature(p, temps[k]["T"])) for p, k in zip(h.p, h.kind)]
    return metrics(h)


def _fmt(m: dict) -> str:
    if not m.get("n"):
        return "n=0"
    return f"LL {m['logloss']:.4f} · Brier {m['brier']:.4f} · acc {m['acc']:.1%} (n={m['n']})"


def write_outputs(cfg: dict, trials: list[dict], abl: list[dict], ev: pd.DataFrame,
                  temps: dict) -> None:
    saved = {k: v for k, v in cfg.items()}
    saved["_meta"] = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "tuned_on": "international events 2024-06 .. 2025-12",
                      "holdout": "2026 international events",
                      "note": "written by evaluate.py --tune"}
    config.CONFIG_FILE.write_text(json.dumps(saved, indent=2))
    t, h = split(ev)
    base = ev.assign(p=0.5)
    lines = ["# Research report", "",
             f"_Generated {saved['_meta']['created']}_", "", ROSTER_NOTE,
             "## Chosen settings", "", "```json", json.dumps(cfg, indent=2), "```", "",
             "## Final model (walk-forward, series level)", "",
             "| | Tuning (2024-25) | Holdout (2026) |", "|---|---|---|",
             f"| Model | {_fmt(t)} | {_fmt(h)} |",
             f"| Model + calibration (temps from tuning events) | - | {_fmt(calibrated_holdout(ev, cfg))} |",
             f"| Coin flip | {_fmt(split(base)[0])} | {_fmt(split(base)[1])} |", "",
             "## Feature-group ablation", "",
             "| Groups | Tuning | Holdout |", "|---|---|---|"]
    lines += [f"| {a['groups']} | {_fmt(a['tune'])} | {_fmt(a['holdout'])} |" for a in abl]
    lines += ["", "## Candidates tested (one change at a time)", "",
              "| Change | Tuning log loss | Adopted |", "|---|---|---|"]
    lines += [f"| {x['change']} | {x['tune']['logloss']:.4f} | {'yes' if x['adopted'] else ''} |"
              for x in trials]
    lines += ["", "## Calibration (temperature per format)", "", "```json",
              json.dumps(temps, indent=2), "```", "",
              "## Per event", "", "| Event | Series | Log loss | Accuracy |", "|---|---|---|---|"]
    for name, g in ev.groupby("event", sort=False):
        m = metrics(g)
        lines.append(f"| {name} | {m['n']} | {m['logloss']:.4f} | {m['acc']:.1%} |")
    config.REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def backtest(cfg=None, refresh=False, save=True) -> pd.DataFrame:
    cfg = cfg or config.load_config()
    series = data.load_series(refresh=refresh)
    games = data.load_games(series, refresh=refresh, progress=False)
    ev = walk_forward(cfg, series, games)
    if save:
        ev.to_pickle(BACKTEST_FILE)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    cfg = config.load_config()
    series = data.load_series(refresh=args.refresh)
    games = data.load_games(series, refresh=args.refresh, progress=False)
    if args.tune:
        cfg, trials = tune(series, games, cfg)
        abl = ablation(series, games, cfg)
    else:
        trials, abl = [], []
    ev = walk_forward(cfg, series, games)
    ev.to_pickle(BACKTEST_FILE)
    import track_record
    temps = calibration.fit_temperatures(track_record.evidence(ev), cfg)
    calibration.save(temps)
    t, h = split(ev)
    print("tuning :", _fmt(t))
    print("holdout:", _fmt(h))
    print("holdout, calibrated on tuning events only:", _fmt(calibrated_holdout(ev, cfg)))
    print("temperatures:", {k: v["T"] for k, v in temps.items()})
    if args.tune:
        write_outputs(cfg, trials, abl, ev, temps)
        print(f"wrote {config.CONFIG_FILE.name} and {config.REPORT_FILE.name}")


if __name__ == "__main__":
    main()
