"""Live loop: notice finished Worlds matches and re-run the model after each one.

``check()`` is cheap (one request: the Worlds schedule). When it sees a match
that finished since the last update, ``refresh()`` runs the full loop:

1. fetch new results and per-game stats (team gold, every player's lane stats)
2. rebuild features walk-forward -> team form, player form and player ratings
   now include the new games; refit the model
3. score the prediction that was logged before that match
4. recalibrate the temperatures with the new evidence
5. log predictions for the next matches and re-simulate the tournament

The update record (which results triggered it, how title odds moved) is saved to
data/last_update.json for the dashboard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
import traceback

import pandas as pd

import calibration
import config
import data
import evaluate
import simulate
import track_record
from predictor import build

STATE_FILE = config.DATA_DIR / "live_state.json"
UPDATE_FILE = config.DATA_DIR / "last_update.json"
BUILT_FILE = config.DATA_DIR / "built.pkl"


def save_built(built) -> None:
    tmp = BUILT_FILE.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(built))
    tmp.replace(BUILT_FILE)


def load_built():
    """The model the updater built last (None if there is none yet)."""
    if not BUILT_FILE.exists():
        return None
    try:
        return pickle.loads(BUILT_FILE.read_bytes())
    except Exception:
        return None


def pairings_key(w: pd.DataFrame) -> str:
    """Fingerprint of the upcoming matches whose teams are known (changes on a draw)."""
    up = w[(w.state != "completed") & (w.team_a != "TBD") & (w.team_b != "TBD")]
    return hashlib.sha1(",".join(sorted(up.match_id + up.team_a + up.team_b)).encode()).hexdigest()


def _read(p, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def worlds_schedule() -> pd.DataFrame:
    rows = data._fetch_league("worlds", "2026-01-01")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.date, utc=True, format="ISO8601")
    return df.sort_values("date").reset_index(drop=True)


def check() -> list[dict]:
    """Worlds matches that finished since the last refresh."""
    w = worlds_schedule()
    seen = set(_read(STATE_FILE, {}).get("completed", []))
    done = w[w.state == "completed"]
    return [{"match_id": r.match_id, "team_a": r.team_a, "team_b": r.team_b,
             "wins_a": int(r.wins_a), "wins_b": int(r.wins_b), "block": r.block}
            for r in done.itertuples() if r.match_id not in seen]


def recalibrate(built) -> dict:
    if evaluate.BACKTEST_FILE.exists():
        bt = pd.read_pickle(evaluate.BACKTEST_FILE)
    else:
        bt = evaluate.walk_forward(built.cfg, built.series, built.games, built.table)
        bt.to_pickle(evaluate.BACKTEST_FILE)
    temps = calibration.fit_temperatures(track_record.evidence(bt), built.cfg)
    calibration.save(temps)
    built.temps = temps
    return temps


def _last_forecast() -> pd.DataFrame | None:
    if not config.FORECAST_LOG.exists():
        return None
    f = pd.read_csv(config.FORECAST_LOG)
    last = f[f.saved_utc == f.saved_utc.max()]
    return last.set_index("team")


def refresh(n_sims: int = 20000, new: list[dict] | None = None):
    """Full update. Returns (built, sim, update_record)."""
    new = new if new is not None else check()
    before = _last_forecast()
    built = build(refresh=True)
    temps = recalibrate(built)
    track_record.log_predictions(built)
    worlds = data.worlds_matches(built.series)
    sim = simulate.run(worlds, built.probs(), n=n_sims)
    done = worlds[worlds.state == "completed"]
    track_record.log_forecast(sim, done.block.iloc[-1] if len(done) else "pre-event")

    scored = track_record.score(built.series)
    verdicts = []
    for m in new:
        row = scored[scored.match_id == m["match_id"]] if "match_id" in scored else scored
        winner = m["team_a"] if m["wins_a"] > m["wins_b"] else m["team_b"]
        v = {**m, "winner": winner}
        if len(row):
            r = row.iloc[0]
            p_winner = r.p_a if r.team_a == winner else 1 - r.p_a
            v.update(predicted_winner=r.fav, p_winner=round(float(p_winner), 3),
                     correct=bool(r.fav == winner))
        verdicts.append(v)
    movers = []
    if before is not None:
        for t in sim.index:
            if t in before.index:
                movers.append({"team": t, "champ_before": float(before.at[t, "champ"]),
                               "champ_after": float(sim.at[t, "champ"]),
                               "adv_before": float(before.at[t, "adv"]),
                               "adv_after": float(sim.at[t, "adv"])})
    record = {"updated_utc": data.now().isoformat(timespec="seconds"),
              "new_results": verdicts, "movers": movers,
              "temperatures": {k: v["T"] for k, v in temps.items()},
              "summary": track_record.summary(scored)}
    save_built(built)
    sim.to_pickle(config.DATA_DIR / "sim.pkl")
    UPDATE_FILE.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    STATE_FILE.write_text(json.dumps({
        "completed": list(worlds[worlds.state == "completed"].match_id),
        "pairings": pairings_key(worlds)}), encoding="utf-8")
    return built, sim, record


def last_update() -> dict | None:
    return _read(UPDATE_FILE, None)


def loop(interval: int = 120, n_sims: int = 20000) -> None:
    """Server updater: re-run the model after every finished match, and whenever
    new pairings are announced (e.g. the Swiss draw), so those get a prediction
    logged before they start."""
    import bootstrap
    print("seeded:", bootstrap.seed(), flush=True)
    first = True
    while True:
        try:
            new = check()
            w = worlds_schedule()
            drawn = pairings_key(w) != _read(STATE_FILE, {}).get("pairings")
            if first or new or drawn:
                why = "start-up" if first else ("new results" if new else "new pairings")
                print(f"{data.now():%Y-%m-%d %H:%M} refresh ({why}): "
                      + ", ".join(f"{m['team_a']} {m['wins_a']}-{m['wins_b']} {m['team_b']}" for m in new),
                      flush=True)
                refresh(n_sims, new)
                first = False
        except Exception:
            traceback.print_exc()
        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Worlds 2026 background updater")
    ap.add_argument("--loop", action="store_true", help="run forever (server)")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("-n", type=int, default=20000)
    a = ap.parse_args()
    if a.loop:
        loop(a.interval, a.n)
    else:
        print(json.dumps(refresh(a.n)[2], indent=2, default=str))
