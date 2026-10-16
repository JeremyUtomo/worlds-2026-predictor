"""Worlds 2026 predictor CLI.

    python main.py update      # the daily loop: fetch -> score -> recalibrate -> predict -> picks
    python main.py pickems     # current Pick'Ems recommendation (no fetch)
    python main.py match GEN T1 --bo 5
    python main.py score       # track record + mistakes report
    python main.py meta        # champion meta tiers and team meta fit
    python evaluate.py --tune  # research run (slow)
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import calibration
import config
import data
import live
import pickems
import simulate
import track_record
from predictor import build

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def _pct(df: pd.DataFrame) -> pd.DataFrame:
    return (df * 100).round(1)


def show_scores(series) -> None:
    sc = track_record.score(series)
    s = track_record.summary(sc)
    if not s.get("scored"):
        print("No scored predictions yet.")
        return
    print(f"Scored {s['scored']} matches: favourite won {s['favourite_won']:.0%}, "
          f"log loss {s['logloss']:.3f} (coin flip {s['coin_flip_logloss']:.3f}), "
          f"Brier {s['brier']:.3f}")
    m = track_record.mistakes(sc)
    if not m.empty:
        print("\nUpsets (favourite lost) and what misled the model:")
        print(m.to_string(index=False))


def show_picks(built, n: int) -> pd.DataFrame:
    worlds = data.worlds_matches(built.series)
    probs = built.probs()
    sim = simulate.run(worlds, probs, n=n)
    print("\nTournament odds (%):")
    print(_pct(sim).to_string())
    print("\nSwiss Pick'Ems (max expected points):")
    print(pickems.swiss_picks(sim).round(3).to_string(index=False))
    order = simulate.load_bracket()
    if order:
        print("\nKnockout bracket picks:")
        print(pickems.bracket_picks(order, lambda a, b: probs(a, b, 5)))
    else:
        print("\nTitle odds (bracket not drawn yet; fill settings/bracket.yaml when it is):")
        print(_pct(sim[["sf", "final", "champ"]].head(8)).to_string())
    return sim


def cmd_update(args) -> None:
    print("Fetching results, refitting team + player form, scoring, recalibrating ...")
    new = live.check()
    built, sim, rec = live.refresh(args.n, new)
    print(f"Data up to {built.table.date.max():%Y-%m-%d %H:%M} UTC, "
          f"{len(built.table)} series, {len(built.games)} games with stats")
    for v in rec["new_results"]:
        verdict = ""
        if "correct" in v:
            verdict = (f" - model had {v['predicted_winner']} "
                       f"({'correct' if v['correct'] else 'WRONG'}, gave {v['winner']} {v['p_winner']:.0%})")
        print(f"New result: {v['team_a']} {v['wins_a']}-{v['wins_b']} {v['team_b']}{verdict}")
    print("\n== Track record ==")
    show_scores(built.series)
    print("\nTemperatures (T<1 sharpen, T>1 flatten):", rec["temperatures"])
    print("\nTournament odds (%):")
    print(_pct(sim).to_string())
    print("\nSwiss Pick'Ems (max expected points):")
    print(pickems.swiss_picks(sim).round(3).to_string(index=False))


def cmd_pickems(args) -> None:
    built = build(refresh=False)
    show_picks(built, args.n)


def cmd_match(args) -> None:
    built = build(refresh=False)
    a, b = args.a.upper(), args.b.upper()
    ex = built.explain(a, b)
    print(f"{a} vs {b}: game {built.p_game(a, b):.1%}, Bo{args.bo} {built.p_match(a, b, args.bo):.1%}")
    print("Logit contribution by group (+ favours", a + "):",
          {k: round(v, 3) for k, v in ex["groups"].items()})
    coef = built.model.coef
    print(pd.DataFrame({"value": ex["features"], "weight": coef,
                        "contribution": coef * pd.Series(ex["features"])}).dropna().round(3).to_string())


def cmd_score(args) -> None:
    show_scores(data.load_series(refresh=False))


def cmd_meta(args) -> None:
    built = build(refresh=False)
    day = built.engine.meta.day(built.now)
    sc = built.engine.meta.scores(day)
    print("Top meta champions (score 0-1):")
    print(sc.head(30).round(3).to_string())
    print("\nStyle profile:", built.engine.meta.style(day))
    rows = []
    for t in config.WORLDS_2026:
        f = built.engine.fit.team(t, built.engine.roster.get(t, []), built.now,
                                  built.engine.form.gd15[t])
        rows.append({"team": t, **f})
    print("\nTeam meta fit:")
    print(pd.DataFrame(rows).set_index("team").sort_values("meta_mastery", ascending=False)
          .round(3).to_string())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("update")
    p.add_argument("-n", type=int, default=20000)
    p.set_defaults(func=cmd_update)
    p = sub.add_parser("pickems")
    p.add_argument("-n", type=int, default=20000)
    p.set_defaults(func=cmd_pickems)
    p = sub.add_parser("match")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--bo", type=int, default=3)
    p.set_defaults(func=cmd_match)
    sub.add_parser("score").set_defaults(func=cmd_score)
    sub.add_parser("meta").set_defaults(func=cmd_meta)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
