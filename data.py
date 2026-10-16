"""Data layer: series results and per-game stats from the public lolesports API.

* ``load_series()``  - one row per series (match) for every league in config.LEAGUES.
* ``load_games()``   - one row per game: champions, players, gold@15, length, winner.

Everything immutable (finished pages, finished games) is cached under data/cache,
so after the first run an update only downloads what is new.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

import config

_session = requests.Session()
_session.headers.update({"x-api-key": config.API_KEY,
                         "User-Agent": "Worlds2026Predictor/0.1 (personal project)"})
_lock = threading.Lock()


def _get(url: str, params: dict | None = None, retries: int = 5) -> dict | None:
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200 and r.content:
            return r.json()
        if r.status_code in (204, 404):
            return None
        time.sleep(1.5 * (attempt + 1) + random.random())
    return None


def _cache_path(*parts: str) -> Path:
    p = config.CACHE_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _write_json(p: Path, obj) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(p)


# --------------------------------------------------------------------------- series
def _schedule_page(league_id: str, token: str | None) -> dict | None:
    params = {"hl": "en-US", "leagueId": league_id}
    if token:
        params["pageToken"] = token
    d = _get(f"{config.API_BASE}/getSchedule", params)
    return d["data"]["schedule"] if d else None


def _fetch_league(slug: str, since: str) -> list[dict]:
    league_id, region = config.LEAGUES[slug]
    events: list[dict] = []
    first = _schedule_page(league_id, None)
    if not first:
        return events
    events += first["events"]
    # newer pages (upcoming matches) are never cached
    tok = first["pages"].get("newer")
    while tok:
        page = _schedule_page(league_id, tok)
        if not page:
            break
        events += page["events"]
        tok = page["pages"].get("newer")
    # older pages: cached once every event on them is finished
    tok = first["pages"].get("older")
    while tok and (not events or min(e["startTime"] for e in events) >= since):
        cp = _cache_path("schedule", slug, hashlib.sha1(tok.encode()).hexdigest()[:16] + ".json")
        page = _read_json(cp)
        if page is None:
            page = _schedule_page(league_id, tok)
            if not page:
                break
            if all(e["state"] == "completed" for e in page["events"]):
                _write_json(cp, page)
        events += page["events"]
        tok = page["pages"].get("older")
    rows = []
    for e in events:
        m = e.get("match")
        if not m or len(m.get("teams", [])) != 2 or e["startTime"] < since:
            continue
        a, b = m["teams"]
        rows.append({
            "match_id": m["id"], "league": slug, "region": region,
            "date": e["startTime"], "block": e.get("blockName") or "",
            "state": e["state"], "bo": (m.get("strategy") or {}).get("count", 1),
            "team_a": a.get("code"), "team_b": b.get("code"),
            "name_a": a.get("name"), "name_b": b.get("name"),
            "wins_a": (a.get("result") or {}).get("gameWins"),
            "wins_b": (b.get("result") or {}).get("gameWins"),
        })
    return rows


def load_series(refresh: bool = True, since: str = config.SERIES_SINCE) -> pd.DataFrame:
    """All series for the configured leagues. Cached to data/series.csv."""
    out = config.DATA_DIR / "series.csv"
    if not refresh and out.exists():
        return _finish_series(pd.read_csv(out, dtype={"match_id": str}))
    rows: list[dict] = []
    with ThreadPoolExecutor(6) as ex:
        for res in ex.map(lambda s: _fetch_league(s, since), config.LEAGUES):
            rows += res
    df = pd.DataFrame(rows).drop_duplicates("match_id")
    df = df.sort_values("date").reset_index(drop=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return _finish_series(df)


def _finish_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
    df["match_id"] = df["match_id"].astype(str)
    return df


# --------------------------------------------------------------------------- games
def _event_details(match_id: str) -> dict | None:
    cp = _cache_path("events", f"{match_id}.json")
    d = _read_json(cp)
    if d is None:
        raw = _get(f"{config.API_BASE}/getEventDetails", {"hl": "en-US", "id": match_id})
        if not raw or not raw.get("data", {}).get("event"):
            return None
        d = raw["data"]["event"]
        if all(g.get("state") in ("completed", "unneeded") for g in d["match"]["games"]):
            _write_json(cp, d)
    return d


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _window(game_id: str, at: datetime | None = None) -> dict | None:
    params = None
    if at is not None:
        at = at.replace(microsecond=0, second=at.second - at.second % 10)
        params = {"startingTime": at.strftime("%Y-%m-%dT%H:%M:%SZ")}
    return _get(f"{config.FEED_BASE}/window/{game_id}", params)


def _team_state(frame: dict, side: str) -> dict:
    t = frame[f"{side}Team"]
    return {"gold": t["totalGold"], "towers": t["towers"], "inhibs": t["inhibitors"],
            "barons": t["barons"], "kills": t["totalKills"], "dragons": len(t["dragons"])}


GAME_VERSION = 2   # v2 adds per-player stats at 15 minutes and at the end


def _players(frame: dict, side: str) -> list[list]:
    """[gold, cs, kills, deaths, assists] per player, in participant order."""
    ps = sorted(frame[f"{side}Team"]["participants"], key=lambda p: p["participantId"])
    return [[p["totalGold"], p["creepScore"], p["kills"], p["deaths"], p["assists"]] for p in ps]


def _add_player_stats(rec: dict, cp: Path) -> dict:
    """Backfill per-player stats for a game cached before they were stored."""
    t0 = _ts(rec["start"])
    w15 = _window(rec["game_id"], t0 + timedelta(minutes=15))
    wend = _window(rec["game_id"], t0 + timedelta(hours=3))
    if not wend or not wend.get("frames"):
        return rec
    if w15 and w15.get("frames"):
        f = w15["frames"][0]
        rec["blue_p15"], rec["red_p15"] = _players(f, "blue"), _players(f, "red")
    end = wend["frames"][-1]
    rec["blue_pend"], rec["red_pend"] = _players(end, "blue"), _players(end, "red")
    rec["v"] = GAME_VERSION
    _write_json(cp, rec)
    return rec


def _game_record(game_id: str) -> dict | None:
    cp = _cache_path("games", f"{game_id}.json")
    rec = _read_json(cp)
    if rec is not None:
        if rec.get("v", 1) < GAME_VERSION:
            rec = _add_player_stats(rec, cp)
        return rec
    w0 = _window(game_id)
    if not w0 or not w0.get("frames"):
        return None
    meta = w0["gameMetadata"]
    start = next((f for f in w0["frames"] if f["blueTeam"]["totalGold"] > 0), w0["frames"][0])
    t0 = _ts(start["rfc460Timestamp"])
    w15 = _window(game_id, t0 + timedelta(minutes=15))
    wend = _window(game_id, t0 + timedelta(hours=3))
    if not wend or not wend.get("frames") or wend["frames"][-1]["gameState"] != "finished":
        return None
    end = wend["frames"][-1]
    b_end, r_end = _team_state(end, "blue"), _team_state(end, "red")

    def score(s):  # whoever took the nexus has the better end state
        return s["inhibs"] * 2 + s["towers"] + s["gold"] / 15000

    rec = {
        "game_id": game_id,
        "patch": ".".join(meta.get("patchVersion", "").split(".")[:2]),
        "start": start["rfc460Timestamp"],
        "minutes": round((_ts(end["rfc460Timestamp"]) - t0).total_seconds() / 60, 2),
        "winner": "blue" if score(b_end) >= score(r_end) else "red",
        "blue_end": b_end, "red_end": r_end,
    }
    if w15 and w15.get("frames"):
        f = w15["frames"][0]
        rec["gd15"] = f["blueTeam"]["totalGold"] - f["redTeam"]["totalGold"]
        rec["blue_15"], rec["red_15"] = _team_state(f, "blue"), _team_state(f, "red")
        rec["blue_p15"], rec["red_p15"] = _players(f, "blue"), _players(f, "red")
    rec["blue_pend"], rec["red_pend"] = _players(end, "blue"), _players(end, "red")
    rec["v"] = GAME_VERSION
    for side in ("blue", "red"):
        md = meta[f"{side}TeamMetadata"]
        rec[f"{side}_team_id"] = md["esportsTeamId"]
        parts = sorted(md["participantMetadata"], key=lambda p: p["participantId"])
        rec[f"{side}_champs"] = [p["championId"] for p in parts]
        rec[f"{side}_players"] = [p.get("esportsPlayerId") for p in parts]
        rec[f"{side}_names"] = [p.get("summonerName", "").strip() for p in parts]
        rec[f"{side}_roles"] = [p.get("role") for p in parts]
    _write_json(cp, rec)
    return rec


def _games_for_match(row) -> list[dict]:
    ev = _event_details(row.match_id)
    if not ev:
        return []
    teams = {t["id"]: t.get("code") for t in ev["match"]["teams"]}
    out = []
    for g in ev["match"]["games"]:
        if g.get("state") != "completed":
            continue
        rec = _game_record(g["id"])
        if not rec:
            continue
        out.append({**rec, "match_id": row.match_id, "number": g["number"],
                    "league": row.league, "region": row.region, "bo": row.bo,
                    "blue": teams.get(rec["blue_team_id"]), "red": teams.get(rec["red_team_id"])})
    return out


def load_games(series: pd.DataFrame, refresh: bool = True, since: str = config.GAMES_SINCE,
               workers: int = 8, progress: bool = True) -> pd.DataFrame:
    out = config.DATA_DIR / "games.pkl"
    if not refresh and out.exists():
        return pd.read_pickle(out)
    todo = series[(series.state == "completed") & (series.date >= pd.Timestamp(since, tz="UTC"))]
    rows: list[dict] = []
    if out.exists():
        # incremental: keep what we have, only fetch series not seen yet
        old = pd.read_pickle(out)
        if len(old):
            have = set(old.match_id)
            todo = todo[~todo.match_id.isin(have)]
            rows = old.drop(columns=["date"]).to_dict("records")
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_games_for_match, r) for r in todo.itertuples()]
        for f in as_completed(futs):
            rows += f.result()
            done += 1
            if progress and done % 200 == 0:
                print(f"  games: {done}/{len(futs)} series", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["start"], utc=True, format="ISO8601")
    df = df.sort_values(["date", "number"]).reset_index(drop=True)
    df.to_pickle(out)
    return df


def worlds_matches(series: pd.DataFrame, year: int = 2026) -> pd.DataFrame:
    w = series[(series.league == "worlds") & (series.date.dt.year == year)]
    return w.sort_values("date").reset_index(drop=True)


def now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


if __name__ == "__main__":
    s = load_series()
    print(f"{len(s)} series, {s.date.min()} .. {s.date.max()}")
    g = load_games(s)
    print(f"{len(g)} games")
