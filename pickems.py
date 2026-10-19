"""Turn simulated probabilities into the Pick'Ems choices with the best expected score."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import config


def swiss_picks(sim: pd.DataFrame, rules: dict = config.PICKEMS) -> pd.DataFrame:
    """Best Swiss Pick'Ems: 8 teams to advance, the first three of them also
    placed on an exact record (3-0, 3-1, 3-2) for bonus points.

    Expected points = advance pts x sum of P(advance) over the 8 picks
                      + bonus x (P(3-0) of the 3-0 pick + P(3-1) of the 3-1 pick
                                 + P(3-2) of the 3-2 pick).
    Searched exhaustively over every choice of the three bonus teams; the other
    picks are the remaining teams most likely to advance.
    """
    slots = rules["swiss_bonus_slots"]
    n_picks, adv_pts, bonus = rules["swiss_picks"], rules["swiss_advance_pts"], rules["swiss_exact_bonus"]
    teams = list(sim.index)
    best = None
    for trio in itertools.permutations(teams, len(slots)):
        rest = sorted((t for t in teams if t not in trio), key=lambda t: -sim.at[t, "adv"])
        rest = rest[: n_picks - len(slots)]
        ev = (adv_pts * sum(sim.at[t, "adv"] for t in (*trio, *rest))
              + bonus * sum(sim.at[t, s] for t, s in zip(trio, slots)))
        if best is None or ev > best[0]:
            best = (ev, trio, rest)
    ev, trio, rest = best
    rows = [{"pick": i + 1, "slot": f"{s} bonus", "team": t, "p_advance": sim.at[t, "adv"],
             "p_exact": sim.at[t, s]} for i, (t, s) in enumerate(zip(trio, slots))]
    rows += [{"pick": len(slots) + i + 1, "slot": "advance", "team": t,
              "p_advance": sim.at[t, "adv"], "p_exact": np.nan} for i, t in enumerate(rest)]
    out = pd.DataFrame(rows)
    out.attrs["expected_points"] = ev
    return out


def bracket_picks(order: list[str], p5, rules: dict = config.PICKEMS) -> dict:
    """Best full bracket (exhaustive over all 128 consistent fills).

    order: 8 teams, QF1 = order[0] v order[1], ...; p5(a, b) = P(a beats b in Bo5).
    """
    def reach(teams):  # exact probability each team wins its sub-bracket
        if len(teams) == 1:
            return {teams[0]: 1.0}
        h = len(teams) // 2
        L, R = reach(teams[:h]), reach(teams[h:])
        out = {}
        for a, pa in L.items():
            out[a] = pa * sum(pb * p5(a, b) for b, pb in R.items())
        for b, pb in R.items():
            out[b] = pb * sum(pa * p5(b, a) for a, pa in L.items())
        return out

    sf_p = {**reach(order[0:2]), **reach(order[2:4]), **reach(order[4:6]), **reach(order[6:8])}
    f_p = {**reach(order[0:4]), **reach(order[4:8])}
    c_p = reach(order)
    pts = rules["ko_pts"]
    best = None
    for qf in itertools.product(*[order[i:i + 2] for i in range(0, 8, 2)]):
        for sf in itertools.product(qf[0:2], qf[2:4]):
            for champ in sf:
                ev = (sum(pts["QF"] * sf_p[t] for t in qf) + sum(pts["SF"] * f_p[t] for t in sf)
                      + pts["F"] * c_p[champ])
                if best is None or ev > best[0]:
                    best = (ev, qf, sf, champ)
    ev, qf, sf, champ = best
    return {"quarterfinal_winners": list(qf), "semifinal_winners": list(sf), "champion": champ,
            "expected_points": round(ev, 1), "p_champion": {t: round(p, 3) for t, p in
                                                            sorted(c_p.items(), key=lambda x: -x[1])}}
