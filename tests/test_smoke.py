"""Smoke tests: run offline in CI before the server deploys a commit."""
from __future__ import annotations

import random

import numpy as np
import pytest

import calibration
import config
import pickems
import simulate
from model import apply_temperature, p_series

TEAMS = list(config.WORLDS_2026)


def _probs(seed: int = 1) -> simulate.Probs:
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1, len(TEAMS))
    g = 1 / (1 + np.exp(-(strength[:, None] - strength[None, :])))
    return simulate.Probs(TEAMS, {bo: p_series(g, bo) for bo in (1, 3, 5)})


@pytest.mark.parametrize("bo", [1, 3, 5])
def test_series_probability_is_symmetric(bo):
    p = np.linspace(0.01, 0.99, 25)
    assert np.allclose(p_series(p, bo) + p_series(1 - p, bo), 1)
    assert np.allclose(apply_temperature(p, 1.7) + apply_temperature(1 - p, 1.7), 1)


def test_tournament_simulation_is_consistent():
    import pandas as pd
    empty = pd.DataFrame(columns=["block", "team_a", "team_b", "state", "wins_a", "wins_b", "bo", "date"])
    sim = simulate.run(empty, _probs(), n=2000, seed=3)
    assert sim.adv.sum() == pytest.approx(8)
    assert sim.champ.sum() == pytest.approx(1)
    assert sim.play_in_win[config.PLAY_IN].sum() == pytest.approx(1)
    swiss = [t for t in TEAMS if t not in config.PLAY_IN]
    records = sim.loc[swiss, ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]].sum(axis=1)
    assert np.allclose(records, 1)


def test_swiss_pairing_avoids_rematches():
    rng = random.Random(0)
    group = TEAMS[:8]
    played = {frozenset(p) for p in zip(group[::2], group[1::2])}
    for _ in range(50):
        pairs = simulate._pair(group, played, config.WORLDS_2026, False, rng)
        assert not any(frozenset(p) in played for p in pairs)


def test_pickems_fill_every_slot_once():
    import pandas as pd
    empty = pd.DataFrame(columns=["block", "team_a", "team_b", "state", "wins_a", "wins_b", "bo", "date"])
    sim = simulate.run(empty, _probs(2), n=1000, seed=5)
    picks = pickems.swiss_picks(sim)
    assert len(picks) == sum(config.PICKEMS["swiss_slots"].values())
    assert picks.team.is_unique
    pr = _probs(2)
    order = TEAMS[:8]
    b = pickems.bracket_picks(order, lambda a, c: pr(a, c, 5))
    assert b["champion"] in b["semifinal_winners"]
    assert set(b["semifinal_winners"]) <= set(b["quarterfinal_winners"])


def test_temperature_shrinks_toward_one_with_little_evidence():
    import pandas as pd
    cfg = config.load_config()
    few = pd.DataFrame({"kind": ["bo1"] * 3, "p": [0.9, 0.9, 0.9], "y": [0, 1, 0]})
    assert calibration.fit_temperatures(few, cfg)["bo1"]["T"] == 1.0
    many = pd.DataFrame({"kind": ["bo1"] * 40, "p": [0.9] * 40, "y": [1, 0] * 20})
    t = calibration.fit_temperatures(many, cfg)["bo1"]
    assert t["learned"] and 1 < t["T"] <= t["raw"]


def test_model_builds_from_seed(tmp_path, monkeypatch):
    """End to end on the bundled seed data (no network)."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    import bootstrap
    import predictor
    assert "games.pkl" in bootstrap.seed()
    b = predictor.build(refresh=False)
    p = b.p_match("GEN", "MVK", 5)
    assert 0.5 < p < 1
    assert b.p_match("MVK", "GEN", 5) == pytest.approx(1 - p, abs=1e-9)
