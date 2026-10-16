"""Turn simulated probabilities into the Pick'Ems choices with the best expected score."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import config


def swiss_picks(sim: pd.DataFrame, rules: dict = config.PICKEMS) -> pd.DataFrame:
    """Assign teams to the 3-0 / 3-1 / 3-2 slots maximising expected points:
    advance points x P(advance) + exact-record bonus x P(that record)."""
    slots = [s for s, n in rules["swiss_slots"].items() for _ in range(n)]
    teams = list(sim.index)
    value = np.array([[rules["swiss_advance_pts"] * sim.at[t, "adv"]
                       + rules["swiss_exact_bonus"] * sim.at[t, s] for s in slots] for t in teams])
    rows, cols = linear_sum_assignment(-value)
    out = pd.DataFrame({"slot": [slots[c] for c in cols], "team": [teams[r] for r in rows],
                        "p_advance": [sim.at[teams[r], "adv"] for r in rows],
                        "p_exact": [sim.at[teams[r], slots[c]] for r, c in zip(rows, cols)],
                        "exp_points": [value[r, c] for r, c in zip(rows, cols)]})
    order = {s: i for i, s in enumerate(rules["swiss_slots"])}
    return out.sort_values(["slot", "p_exact"], key=lambda c: c.map(order) if c.name == "slot" else -c) \
              .reset_index(drop=True)


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
