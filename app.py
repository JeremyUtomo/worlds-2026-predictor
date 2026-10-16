"""Streamlit dashboard: predictions, Pick'Ems, track record, meta.

    streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import calibration
import config
import data
import evaluate
import live
import pickems
import simulate
import track_record
from predictor import build

st.set_page_config(page_title="Worlds 2026 Predictor", page_icon="🏆", layout="wide")


SERVER = config.MODE == "server"   # on the website a separate updater refreshes the model


@st.cache_resource
def _store() -> dict:
    return {}


def stamp() -> str:
    """Changes every time the model is re-run (by the updater or a local refresh)."""
    u = live.last_update()
    return u["updated_utc"] if u else "none"


def get_built(version: str):
    store = _store()
    if store.get("version") != version or "built" not in store:
        with st.spinner("Loading model ..."):
            b = live.load_built()
            if b is None:
                b = build(refresh=False)
        store.update(built=b, version=version)
    return store["built"]


@st.cache_data(show_spinner="Simulating the tournament ...")
def get_sim(version: str, n: int):
    b = get_built(version)
    return simulate.run(data.worlds_matches(b.series), b.probs(), n=n)


@st.cache_data(ttl=60, show_spinner=False)
def schedule(version: str) -> pd.DataFrame:
    return live.worlds_schedule()


def run_refresh(new: list[dict], n: int) -> None:
    label = ", ".join(f"{m['team_a']} {m['wins_a']}-{m['wins_b']} {m['team_b']}" for m in new)         or "manual refresh"
    with st.spinner(f"New result ({label}) - refitting team + player form, scoring, recalibrating ..."):
        live.refresh(n, new)
    st.cache_data.clear()
    st.toast(f"Model updated after {label}")


with st.sidebar:
    st.title("Worlds 2026")
    page = st.radio("Page", ["Upcoming matches", "Predictions", "Pick'Ems", "Track record", "Meta"])
    n_sims = st.select_slider("Simulations", [5000, 20000, 50000], value=20000)
    if SERVER:
        auto = False
        st.caption("Updates automatically after every finished match.")
    else:
        auto = st.toggle("Auto-update after each match", value=True,
                         help="Checks the Worlds schedule every 2 minutes; when a match finishes "
                              "the model is re-run with the new team and player form.")
        if st.button("Fetch latest results"):
            run_refresh(live.check(), n_sims)


@st.fragment(run_every="60s" if SERVER else "120s")
def watcher():
    if SERVER:
        # the updater re-ran the model: reload the page with the new version
        if stamp() != st.session_state.get("seen_stamp"):
            st.rerun(scope="app")
        return
    if not auto:
        return
    try:
        new = live.check()
    except Exception as e:  # network hiccup: try again next tick
        st.sidebar.caption(f"Schedule check failed: {e}")
        return
    if new:
        st.session_state.pending = new
        st.rerun(scope="app")


# process finished matches first, then start the timer (otherwise the watcher would
# see the same unprocessed results and rerun the page before they are handled)
if not SERVER and (st.session_state.pop("pending", None)
                   or (auto and st.session_state.get("first_check", True))):
    _new = live.check()
    if _new:
        run_refresh(_new, n_sims)
st.session_state.first_check = False
st.session_state.version = st.session_state.seen_stamp = stamp()
with st.sidebar:
    watcher()

built = get_built(st.session_state.version)
built.temps = calibration.load()
st.sidebar.caption(f"Data up to {built.table.date.max():%Y-%m-%d %H:%M} UTC · "
                   f"{len(built.games):,} games with stats")
pct = lambda df: (df * 100).round(1)


def team_form(team: str) -> dict:
    eng, now = built.engine, built.now
    s = built.series
    done = s[(s.state == "completed") & ((s.team_a == team) | (s.team_b == team))].sort_values("date")
    last5 = "".join("W" if (r.team_a == team) == (r.wins_a > r.wins_b) else "L"
                    for r in done.tail(5).itertuples())
    return {"Form rating": round(eng.form.rating(team, now)),
            "Recent game win %": f"{eng.form.win[team]:.0%}",
            "Gold @15": f"{eng.form.gd15[team]:+.0f}",
            "Last 5 series": last5 or "-",
            "Roster continuity": f"{eng.rost.continuity(team, eng.lineup(team), now):.0%}"}


def player_table(team: str) -> pd.DataFrame:
    eng = built.engine
    roster = eng.lineup(team)
    base = eng.history.region_rating.get(eng.history.region_of(team), 0.0)
    rows = eng.pform.card(roster)
    for r, p in zip(rows, roster):
        r["rating"] = round(base + eng.rost._off(p, team))
    return pd.DataFrame(rows)


def reasons(a: str, b: str) -> str:
    groups = built.explain(a, b)["groups"]
    names = {"form": "team form", "player_form": "player form", "meta_fit": "meta fit",
             "history": "history", "roster": "roster"}
    parts = [f"{names.get(g, g)} favours {a if v > 0 else b}" for g, v in
             sorted(groups.items(), key=lambda x: -abs(x[1])) if abs(v) > 0.03]
    return "; ".join(parts) or "evenly matched on every factor"


if page == "Upcoming matches":
    st.header("Upcoming matches")
    upd = live.last_update()
    if upd:
        with st.container(border=True):
            st.markdown(f"**Last model update:** {upd['updated_utc'].replace('T', ' ')[:16]} UTC")
            for v in upd.get("new_results", []):
                line = f"{v['team_a']} **{v['wins_a']}-{v['wins_b']}** {v['team_b']}"
                if "correct" in v:
                    line += (f" - predicted {v['predicted_winner']} "
                             f"{'✅' if v['correct'] else '❌'} (gave {v['winner']} {v['p_winner']:.0%})")
                st.markdown(line)
            mv = pd.DataFrame(upd.get("movers", []))
            if len(mv):
                mv["Δ title"] = (mv.champ_after - mv.champ_before) * 100
                mv["Δ advance"] = (mv.adv_after - mv.adv_before) * 100
                mv = mv.reindex(mv["Δ title"].abs().add(mv["Δ advance"].abs())
                                .sort_values(ascending=False).index).head(5)
                st.caption("Biggest odds changes from that update (percentage points)")
                st.dataframe(mv[["team", "Δ title", "Δ advance"]].round(1), hide_index=True)
    sched = schedule(st.session_state.version)
    up = sched[sched.state.isin(["unstarted", "inProgress"])]
    if up.empty:
        st.info("No upcoming Worlds matches.")
    if not SERVER:
        track_record.log_predictions(built)
    tbd = up[(up.team_a == "TBD") | (up.team_b == "TBD")]
    for blk, grp in tbd.groupby("block", sort=False):
        st.caption(f"{blk}: {len(grp)} match(es) from {grp.date.min():%a %d %b %H:%M} UTC waiting for "
                   "their teams - predictions appear as soon as both teams are known.")
    for r in up.drop(tbd.index).itertuples():
        when = r.date.strftime("%a %d %b %H:%M UTC")
        live_tag = f" · 🔴 LIVE {int(r.wins_a or 0)}-{int(r.wins_b or 0)}" if r.state == "inProgress" else ""
        with st.container(border=True):
            st.caption(f"{when} · {r.block} · Bo{int(r.bo)}{live_tag}")
            if "TBD" in (r.team_a, r.team_b):
                st.markdown(f"**{r.team_a}** vs **{r.team_b}** - opponent not decided yet; "
                            "the prediction appears as soon as both teams are known.")
                continue
            a, b, bo = r.team_a, r.team_b, int(r.bo)
            p = built.p_match(a, b, bo)
            c1, c2, c3 = st.columns([2, 5, 2])
            c1.metric(a, f"{p:.0%}")
            c3.metric(b, f"{1 - p:.0%}")
            c2.progress(p, text=f"Predicted winner: **{a if p >= 0.5 else b}** "
                                f"(per game {built.p_game(a, b):.0%} / {1 - built.p_game(a, b):.0%})")
            st.caption("Why: " + reasons(a, b))
            with st.expander("Team and player form"):
                for tab, t in zip(st.tabs([a, b]), (a, b)):
                    tab.dataframe(pd.DataFrame([team_form(t)]), hide_index=True, width="stretch")
                    tab.dataframe(player_table(t), hide_index=True, width="stretch")
    st.subheader("Recent results vs prediction")
    sc = track_record.score(built.series)
    if len(sc) and "status" in sc:
        done = sc[sc.status == "Scored"].sort_values("date", ascending=False).head(10)
        if len(done):
            res = done.assign(result=lambda d: d.apply(
                lambda x: f"{x.team_a} {int(x.wins_a if x.res_a == x.team_a else x.wins_b)}-"
                          f"{int(x.wins_b if x.res_a == x.team_a else x.wins_a)} {x.team_b}", axis=1),
                predicted=lambda d: d.fav + " " + (d[["p_a"]].max(axis=1).where(d.p_a >= .5, 1 - d.p_a) * 100)
                .round(0).astype(int).astype(str) + "%",
                hit=lambda d: d.fav_won.map({1.0: "✅", 0.0: "❌"}))
            st.dataframe(res[["date", "block", "result", "predicted", "hit"]], hide_index=True)
        else:
            st.caption("No logged prediction has a result yet.")
    else:
        st.caption("No logged predictions yet.")

elif page == "Predictions":
    st.header("Tournament odds")
    sim = get_sim(st.session_state.version, n_sims)
    show = sim[sim.adv > 0]
    st.dataframe(pct(show).style.format("{:.1f}").background_gradient(cmap="Blues", subset=["adv", "sf", "champ"]),
                 width="stretch")
    fig = px.bar(show.reset_index(), x="index", y="champ", labels={"index": "", "champ": "P(champion)"})
    fig.update_layout(yaxis_tickformat=".0%", height=320)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Head to head")
    c1, c2, c3 = st.columns(3)
    teams = list(config.WORLDS_2026)
    a = c1.selectbox("Team A", teams, index=0)
    b = c2.selectbox("Team B", teams, index=2)
    bo = c3.selectbox("Format", [1, 3, 5], index=2)
    if a != b:
        ex = built.explain(a, b)
        st.metric(f"{a} beats {b} (Bo{bo})", f"{built.p_match(a, b, bo):.1%}",
                  help=f"Per game {built.p_game(a, b):.1%}")
        coef = built.model.coef
        contrib = (coef * pd.Series(ex["features"])).reindex(coef.index)
        fig = px.bar(contrib.reset_index(), x=0, y="index", orientation="h",
                     labels={"0": f"logit contribution (+ favours {a})", "index": ""})
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

elif page == "Pick'Ems":
    sim = get_sim(st.session_state.version, n_sims)
    st.header("Swiss picks")
    st.caption("Slots chosen to maximise expected points: advance points × P(advance) "
               "+ exact-record bonus × P(record). Edit scoring in config.PICKEMS.")
    st.dataframe(pickems.swiss_picks(sim).round(3), width="stretch")
    st.header("Knockout bracket")
    order = simulate.load_bracket()
    if order:
        probs = built.probs()
        st.json(pickems.bracket_picks(order, lambda x, y: probs(x, y, 5)))
    else:
        st.info("Bracket not drawn yet. Add it to settings/bracket.yaml when Riot publishes it.")
        st.dataframe(pct(sim[["sf", "final", "champ"]].head(8)), width="stretch")

elif page == "Track record":
    st.header("Live Worlds predictions")
    sc = track_record.score(built.series)
    s = track_record.summary(sc)
    if s.get("scored"):
        c = st.columns(4)
        c[0].metric("Scored matches", s["scored"])
        c[1].metric("Favourite won", f"{s['favourite_won']:.0%}")
        c[2].metric("Log loss", f"{s['logloss']:.3f}", help=f"Coin flip = {s['coin_flip_logloss']:.3f}")
        c[3].metric("Brier", f"{s['brier']:.3f}")
        st.subheader("Mistakes report")
        st.caption("Upsets, and which feature groups pushed toward the team that lost.")
        st.dataframe(track_record.mistakes(sc), width="stretch")
    if not sc.empty:
        st.dataframe(sc[["date", "block", "team_a", "team_b", "p_a", "status", "fav_won", "logloss"]],
                     width="stretch")
    else:
        st.info("Nothing logged yet - a prediction is saved for every match as soon as both "
                "teams are known, and frozen when the match starts.")
    st.header("Backtest (international events since mid-2024)")
    if evaluate.BACKTEST_FILE.exists():
        bt = pd.read_pickle(evaluate.BACKTEST_FILE)
        rows = [{"event": e, **evaluate.metrics(g)} for e, g in bt.groupby("event", sort=False)]
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch")
        st.subheader("Calibration")
        temps = calibration.load()
        st.write({k: v.get("T") for k, v in temps.items()})
        ct = calibration.calibration_table(track_record.evidence(bt), temps).reset_index()
        fig = px.scatter(ct, x="predicted", y="observed", size="n", range_x=[0.5, 1], range_y=[0.3, 1])
        fig.add_shape(type="line", x0=0.5, y0=0.5, x1=1, y1=1, line=dict(dash="dash"))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Run `python evaluate.py` to create the backtest.")
    if config.REPORT_FILE.exists():
        with st.expander("Research report"):
            st.markdown(config.REPORT_FILE.read_text(encoding="utf-8"))

else:
    eng = built.engine
    day = eng.meta.day(built.now)
    st.header("Champion meta")
    st.caption(f"Pick share × win rate over the last {built.cfg['meta']['window_days']} days, "
               "international games weighted up, adjusted for patch changes.")
    sc = eng.meta.scores(day).head(30).rename("meta score")
    fig = px.bar(sc.reset_index(), x="champ", y="meta score", labels={"champ": ""})
    st.plotly_chart(fig, width="stretch")
    st.write("Style profile:", eng.meta.style(day))
    st.header("Team meta fit")
    rows = [{"team": t, **eng.fit.team(t, eng.roster.get(t, []), built.now, eng.form.gd15[t])}
            for t in config.WORLDS_2026]
    st.dataframe(pd.DataFrame(rows).set_index("team").sort_values("meta_mastery", ascending=False)
                 .round(3), width="stretch")
