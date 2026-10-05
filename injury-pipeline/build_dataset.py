#!/usr/bin/env python3
"""
build_dataset.py  -  nflverse + Sleeper  ->  dataset.json for Injury Ledger

Sources (all free, no auth):
  weekly rosters  https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{season}.parquet
  injury reports  https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet
  player stats    https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{season}.parquet
                  (fallback: .../stats_player_week_{season}.parquet - newer naming)
  schedules       https://github.com/nflverse/nfldata/raw/master/data/games.csv
  player ids      https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet
  live status     https://api.sleeper.app/v1/players/nfl
  preseason ADP   https://api.myfantasyleague.com/{season}/export?TYPE=adp&PERIOD=JULY...
                  (drafts completed before preseason game 1; overall rank across all positions)
  id crosswalk    https://github.com/dynastyprocess/data/raw/master/files/db_playerids.csv (mfl_id -> gsis_id)

Output (default ./data):
  dataset.json              everything the HTML app needs (meta + players[])
  players/<sleeper_id>.json one file per player (for lazy loading later)
  player_seasons.csv        flat table for spreadsheets / BI

Usage:
  python build_dataset.py --start 2016 --end 2026 --scoring half --out ./data
  python build_dataset.py --no-sleeper          # skip live status
  python build_dataset.py --refresh             # ignore local cache

Weekly status mapping (per player, per regular-season week):
  team has no game ................ BYE   (excluded)
  week not yet played ............. FUT   (excluded)
  player not on any roster ........ NR    (excluded - unsigned / retired)
  roster status SUS/EXE ........... SUS   (excluded - not an injury)
  roster status RES/PUP/NON/IR .... IR    (missed, listed)
  report "Out" / "Doubtful" DNP ... O / D (missed, listed)
  report "Questionable" & played .. Q     (played, listed)
  report "Questionable" & DNP ..... O     (missed, listed)
  INA / no stats, no report ....... DNP   (missed, NOT listed - healthy scratch
                                            or zero-stat game)
  played, no report ............... H
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
URLS = {
    "rosters":   NFLVERSE + "/weekly_rosters/roster_weekly_{season}.parquet",
    "injuries":  NFLVERSE + "/injuries/injuries_{season}.parquet",
    "stats":     NFLVERSE + "/player_stats/player_stats_{season}.parquet",
    "stats_new": NFLVERSE + "/player_stats/stats_player_week_{season}.parquet",
    "players":   NFLVERSE + "/players/players.parquet",
    "schedule":  "https://github.com/nflverse/nfldata/raw/master/data/games.csv",
    "sleeper":   "https://api.sleeper.app/v1/players/nfl",
    "adp":       "https://api.myfantasyleague.com/{season}/export?TYPE=adp&PERIOD={period}&FCOUNT={fcount}&IS_PPR={ppr}&IS_KEEPER=N&IS_MOCK=0&JSON=1",
    "playerids": "https://github.com/dynastyprocess/data/raw/master/files/db_playerids.csv",
}
# MFL ADP periods: JULY = drafts in July (strictly before any preseason game),
# AUG1 = first half of August (preseason week 1 usually falls in it), AUG15, START, ALL ...
FANTASY_POS = {"QB", "RB", "WR", "TE"}
TEAM_FIX = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "JAC": "JAX", "WSH": "WAS"}
POS_BASE_PPG = {"QB": 17.0, "RB": 11.0, "WR": 11.0, "TE": 7.5}
RESERVE = ("RES", "PUP", "NON", "IR")

# body-part classification used by the app for soft-tissue vs structural views
SOFT = ("hamstring", "calf", "groin", "quad", "oblique", "thigh", "hip flexor", "soft")
HEAD = ("concussion", "head", "neck")
STRUCT = ("acl", "achilles", "lisfranc", "fracture", "broken", "collarbone", "clavicle",
          "meniscus", "back", "spine", "foot", "mcl", "pcl", "torn")


# ---------------------------------------------------------------- fetching ---
def fetch(url, cache, refresh, kind):
    cache.mkdir(parents=True, exist_ok=True)
    if kind == "json":
        name = "sleeper_players.json" if "sleeper" in url else "adp_" + url.split("/")[3] + "_" + url.split("PERIOD=")[1].split("&")[0] + ".json"
    else:
        name = url.split("/")[-1]
    local = cache / name
    fresh = local.exists() and not refresh
    if kind == "json" and "sleeper" in url and fresh and time.time() - local.stat().st_mtime > 3600:
        fresh = False  # live status: max 1h old
    if not fresh:
        if requests is None:
            raise SystemExit("python package 'requests' is missing")
        print(f"  GET {url}")
        r = requests.get(url, timeout=120)
        if r.status_code == 404:
            raise FileNotFoundError(url)
        r.raise_for_status()
        local.write_bytes(r.content)
    if kind == "parquet":
        return pd.read_parquet(local)
    if kind == "csv":
        return pd.read_csv(local, low_memory=False)
    return json.loads(local.read_text())


def is_na(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def norm_team(t):
    if is_na(t):
        return None
    t = str(t).upper()
    return TEAM_FIX.get(t, t)


def first_col(df, *names, default="__raise__"):
    for n in names:
        if n in df.columns:
            return df[n]
    if default == "__raise__":
        raise KeyError(f"none of {names} in columns {list(df.columns)[:25]}...")
    return pd.Series([default] * len(df), index=df.index, dtype="object")


def classify(body):
    if not body:
        return "unknown"
    b = body.lower()
    if any(k in b for k in HEAD):
        return "head"
    if any(k in b for k in SOFT):
        return "soft"
    if any(k in b for k in STRUCT):
        return "structural"
    return "joint"


def clean_id(v):
    if is_na(v):
        return None
    s = str(v)
    return s[:-2] if s.endswith(".0") else s


# ------------------------------------------------------------ transforms ----
def season_frame(season, rosters, injuries, stats, schedule, scoring):
    """Index everything for one season by (gsis_id, week)."""
    # ----- schedule: weeks, byes, completed weeks
    sch = schedule[(schedule["season"] == season) & (schedule["game_type"] == "REG")].copy()
    if sch.empty:
        raise ValueError(f"no schedule rows for {season}")
    n_weeks = int(sch["week"].max())
    team_weeks = defaultdict(set)
    for _, g in sch.iterrows():
        team_weeks[norm_team(g["home_team"])].add(int(g["week"]))
        team_weeks[norm_team(g["away_team"])].add(int(g["week"]))
    done = sch["result"].notna()
    completed = int(sch.loc[done, "week"].max()) if done.any() else 0

    # ----- weekly rosters: backbone
    ro = rosters.copy()
    ro = ro[first_col(ro, "game_type", "season_type", default="REG").fillna("REG").eq("REG")]
    ro["gsis_id"] = first_col(ro, "gsis_id", "player_id")
    ro["team_n"] = first_col(ro, "team").map(norm_team)
    ro["week"] = ro["week"].astype(int)
    ro["status_r"] = first_col(ro, "status", default="ACT").fillna("ACT").astype(str).str.upper()
    ro = ro[ro["gsis_id"].notna()]
    roster_idx = {(g, w): (t, s) for g, w, t, s in zip(ro["gsis_id"], ro["week"], ro["team_n"], ro["status_r"])}
    roster_meta = {}
    name_col = first_col(ro, "full_name", "player_name", default="")
    sl_col = first_col(ro, "sleeper_id", default=None)
    rk_col = first_col(ro, "rookie_year", "rookie_season", "entry_year", default=None)
    ro = ro.assign(_name=name_col, _sl=sl_col, _rk=rk_col).sort_values("week")
    for g, grp in ro.groupby("gsis_id"):
        last = grp.iloc[-1]
        sl = grp["_sl"].dropna()
        rk = grp["_rk"].dropna()
        roster_meta[g] = {
            "name": str(last["_name"] or ""),
            "pos": str(first_col(grp, "position", default="").iloc[-1] or ""),
            "team": last["team_n"],
            "sleeper_id": clean_id(sl.iloc[-1]) if len(sl) else None,
            "rookie": int(rk.iloc[0]) if len(rk) else None,
        }

    # ----- injury reports
    inj = injuries.copy()
    inj = inj[first_col(inj, "game_type", "season_type", default="REG").fillna("REG").eq("REG")]
    inj["week"] = inj["week"].astype(int)
    rep = {}
    for r in inj.to_dict("records"):
        status = r.get("report_status")
        body = r.get("report_primary_injury")
        if not isinstance(body, str):
            body = r.get("practice_primary_injury")
        if not isinstance(status, str) and not isinstance(body, str):
            continue
        rep[(r["gsis_id"], int(r["week"]))] = {
            "status": status.strip().title() if isinstance(status, str) else None,
            "body": body.strip().title() if isinstance(body, str) else None,
        }

    # ----- stats: played + points
    st = stats.copy()
    st = st[first_col(st, "season_type", "game_type", default="REG").fillna("REG").eq("REG")]
    st["gsis_id"] = first_col(st, "player_id", "gsis_id")
    st["week"] = st["week"].astype(int)
    std = pd.to_numeric(first_col(st, "fantasy_points", default=0.0), errors="coerce").fillna(0.0)
    ppr = pd.to_numeric(first_col(st, "fantasy_points_ppr", default=None), errors="coerce")
    ppr = ppr.fillna(std) if ppr.notna().any() else std
    pts = {"std": std, "ppr": ppr, "half": (std + ppr) / 2}[scoring]
    played = {(g, w): float(p) for g, w, p in zip(st["gsis_id"], st["week"], pts)}

    return {"n_weeks": n_weeks, "team_weeks": team_weeks, "completed": completed,
            "roster_idx": roster_idx, "roster_meta": roster_meta, "rep": rep, "played": played}


REPORT_TO_STATUS = {"Out": "O", "Doubtful": "D", "Questionable": "Q"}


def build_player_season(gsis, season, F, is_current):
    n_weeks, completed = F["n_weeks"], F["completed"]
    weeks, injuries = [], []
    team_prev, ever_on_roster = None, False
    for w in range(1, n_weeks + 1):
        key = (gsis, w)
        roster = F["roster_idx"].get(key)
        team = roster[0] if roster else team_prev
        rstatus = roster[1] if roster else None
        rep = F["rep"].get(key)
        pts = F["played"].get(key)
        body = rep["body"] if rep else None
        rstat = rep["status"] if rep else None
        row = {"wk": w, "status": "H", "played": False, "inj": None}

        if is_current and w > completed:
            row["status"] = "FUT"
            if w == completed + 1 and (rep or rstatus in RESERVE):
                nxt = "IR" if rstatus in RESERVE else REPORT_TO_STATUS.get(rstat, "Q")
                row["next"] = {"status": nxt, "inj": body}
        elif roster is None and pts is None and rep is None:
            row["status"] = "NR"
        elif team and w not in F["team_weeks"].get(team, set()):
            row["status"] = "BYE"
        elif rstatus in ("SUS", "EXE"):
            row["status"] = "SUS"
        elif rstatus in RESERVE and pts is None:
            row["status"] = "IR"
            row["inj"] = body or (injuries[-1]["type"] if injuries else "Reserve list")
        elif pts is not None:
            row["played"] = True
            if rstat in ("Questionable", "Doubtful"):
                row["status"], row["inj"] = "Q", body
        else:  # did not play
            if rstat == "Doubtful":
                row["status"], row["inj"] = "D", body
            elif rstat in ("Out", "Questionable") or body:
                row["status"], row["inj"] = "O", body
            else:
                row["status"] = "DNP"
        if roster:
            ever_on_roster = True
        team_prev = team
        weeks.append(row)

        # injury events: consecutive listed weeks with the same body part
        if row["status"] in ("Q", "D", "O", "IR") and row["inj"]:
            last = injuries[-1] if injuries else None
            missed = 0 if row["played"] else 1
            if last and last["_end"] == w - 1 and last["type"] == row["inj"]:
                last["_end"] = w
                last["listed"] += 1
                last["missed"] += missed
            else:
                injuries.append({"week": w, "type": row["inj"], "cls": classify(row["inj"]),
                                 "sev": None, "missed": missed, "listed": 1, "_end": w})

    if not ever_on_roster and not any(r["played"] for r in weeks):
        return None  # no NFL season for this player

    for e in injuries:
        e["sev"] = "major" if e["missed"] >= 4 else "mod" if e["missed"] >= 1 else "minor"
        del e["_end"]

    counted = [r for r in weeks if r["status"] not in ("BYE", "FUT", "NR", "SUS")]
    gp = sum(1 for r in counted if r["played"])
    gm = len(counted) - gp
    pts_played = [F["played"][(gsis, r["wk"])] for r in weeks if r["played"]]
    ppg = round(sum(pts_played) / len(pts_played), 2) if pts_played else None
    listed = {k: sum(1 for r in counted if r["status"] == k) for k in ("Q", "D", "O", "IR")}
    return {
        "season": season, "nWeeks": n_weeks,
        "bye": next((r["wk"] for r in weeks if r["status"] == "BYE"), None),
        "ppg": ppg, "weeks": weeks, "injuries": injuries,
        "gp": gp, "gm": gm, "gamesPossible": len(counted), "listed": listed,
        "listedTotal": sum(listed.values()),
        "listedWeighted": listed["Q"] * 0.25 + listed["D"] * 0.5 + listed["O"] + listed["IR"],
        "injCount": len(injuries), "hadIR": listed["IR"] > 0,
        "ptsLost": round(gm * ppg, 1) if ppg is not None else None,
    }


def finalize_player(p, sleeper):
    """Fill ppg gaps, points lost, current status."""
    seasons = p["seasons"]
    years = sorted(seasons)
    for i, y in enumerate(years):
        s = seasons[y]
        if s["ppg"] is None:
            cand = [seasons[z]["ppg"] for z in years[:i][::-1] + years[i + 1:] if seasons[z]["ppg"] is not None]
            s["ppg"] = cand[0] if cand else POS_BASE_PPG.get(p["pos"], 8.0)
            s["ppgEstimated"] = True
            s["ptsLost"] = round(s["gm"] * s["ppg"], 1)
    cur = {"status": "H", "inj": None}
    if sleeper:
        st = (sleeper.get("injury_status") or "").upper()
        m = {"IR": "IR", "OUT": "O", "DOUBTFUL": "D", "QUESTIONABLE": "Q", "PUP": "IR", "NA": "IR", "COV": "O"}
        if st in m:
            cur = {"status": m[st], "inj": sleeper.get("injury_body_part") or sleeper.get("injury_notes"),
                   "since": sleeper.get("injury_start_date")}
        if sleeper.get("team"):
            p["team"] = norm_team(sleeper["team"])
    elif years:
        last = seasons[years[-1]]
        nxt = next((r.get("next") for r in last["weeks"] if r.get("next")), None)
        if nxt:
            cur = nxt
    p["current"] = cur


# -------------------------------------------------------------------- ADP ----
def load_adp(season, cache, refresh, period, fcount, ppr_flag, mfl_to_gsis, pos_of):
    """Return gsis_id -> {pick, rank, posRank, drafts, period} from MFL preseason ADP."""
    url = URLS["adp"].format(season=season, period=period, fcount=fcount, ppr=ppr_flag)
    try:
        raw = fetch(url, cache, refresh, "json")
    except Exception as e:
        print(f"  ! ADP unavailable for {season}: {e}")
        return {}
    players = ((raw.get("adp") or {}).get("players") or {}).get("player") or []
    if isinstance(players, dict):
        players = [players]
    rows = []
    for r in players:
        try:
            rows.append((float(r["averagePick"]), str(r["id"]), int(float(r.get("draftsSelectedIn", 0) or 0))))
        except (KeyError, ValueError, TypeError):
            continue
    rows.sort()
    out, pos_counter = {}, defaultdict(int)
    rank = 0
    for pick, mfl_id, drafts in rows:
        rank += 1
        gsis = mfl_to_gsis.get(mfl_id)
        pos = pos_of.get(gsis) if gsis else None
        if pos:
            pos_counter[pos] += 1
        if gsis:
            out[gsis] = {"pick": round(pick, 1), "rank": rank, "posRank": f"{pos}{pos_counter[pos]}" if pos else None,
                         "drafts": drafts, "period": period, "source": "MFL"}
    print(f"  ADP {season}/{period}: {len(rows)} players, {len(out)} mapped to gsis ids, {raw.get('adp', {}).get('totalDrafts', '?')} drafts")
    return out


def load_mfl_crosswalk(cache, refresh):
    try:
        df = fetch(URLS["playerids"], cache, refresh, "csv")
    except Exception as e:
        print(f"  ! player id crosswalk unavailable: {e}")
        return {}
    df = df[df["mfl_id"].notna() & df["gsis_id"].notna()]
    return {clean_id(m): str(g) for m, g in zip(df["mfl_id"], df["gsis_id"])}


# ------------------------------------------------------------------ main ----
def load_player_index(players):
    players = players.copy()
    players["gsis_id"] = first_col(players, "gsis_id")
    rk = "rookie_season" if "rookie_season" in players.columns else "rookie_year"
    idx = {}
    for r in players[players["gsis_id"].notna()].to_dict("records"):
        idx[r["gsis_id"]] = {
            "name": r.get("display_name") or r.get("full_name"),
            "pos": r.get("position"),
            "team": norm_team(r.get("latest_team") or r.get("team_abbr") or r.get("team")),
            "sleeper_id": clean_id(r.get("sleeper_id")),
            "espn_id": clean_id(r.get("espn_id")), "yahoo_id": clean_id(r.get("yahoo_id")),
            "rookie": None if is_na(r.get(rk)) else int(r.get(rk)),
        }
    return idx


def run(args):
    cache, out = Path(args.cache), Path(args.out)
    print("Loading reference data")
    schedule = fetch(URLS["schedule"], cache, args.refresh, "csv")
    pl_idx = load_player_index(fetch(URLS["players"], cache, args.refresh, "parquet"))
    sleeper = {}
    if not args.no_sleeper:
        try:
            raw = fetch(URLS["sleeper"], cache, args.refresh, "json")
            sleeper = {v.get("gsis_id"): dict(v, sleeper_id=k) for k, v in raw.items() if v.get("gsis_id")}
            print(f"  Sleeper: {len(raw)} players, {len(sleeper)} with gsis_id")
        except Exception as e:  # live status is optional
            print(f"  ! Sleeper status unavailable: {e}")

    mfl_to_gsis = {} if args.no_adp else load_mfl_crosswalk(cache, args.refresh)
    pos_of = {g: v.get("pos") for g, v in pl_idx.items()}
    ppr_flag = {"ppr": 1, "std": 0, "half": -1}[args.scoring]
    players_out = {}
    completed_current = 0
    for season in range(args.start, args.end + 1):
        print(f"Season {season}")
        try:
            rosters = fetch(URLS["rosters"].format(season=season), cache, args.refresh, "parquet")
            injuries = fetch(URLS["injuries"].format(season=season), cache, args.refresh, "parquet")
            try:
                stats = fetch(URLS["stats"].format(season=season), cache, args.refresh, "parquet")
            except FileNotFoundError:
                stats = fetch(URLS["stats_new"].format(season=season), cache, args.refresh, "parquet")
        except FileNotFoundError as e:
            print(f"  ! missing release asset, skipping season: {e}")
            continue
        F = season_frame(season, rosters, injuries, stats, schedule, args.scoring)
        adp = {} if args.no_adp else load_adp(season, cache, args.refresh, args.adp_period, args.adp_teams, ppr_flag, mfl_to_gsis, pos_of)
        is_current = season == args.end
        if is_current:
            completed_current = F["completed"]
        season_pts = defaultdict(float)
        for (g, _w), p in F["played"].items():
            season_pts[g] += p
        for gsis, meta in F["roster_meta"].items():
            if meta["pos"] not in FANTASY_POS:
                continue
            relevant = season_pts.get(gsis, 0.0) >= args.min_points or gsis in players_out
            if not relevant and gsis in adp and adp[gsis]["rank"] <= args.adp_keep:
                relevant = True  # drafted early but scored nothing (e.g. preseason injury) - exactly what we track
            if not relevant and is_current:
                relevant = (sleeper.get(gsis, {}).get("search_rank") or 9999) <= args.sleeper_rank
            if not relevant:
                continue
            ps = build_player_season(gsis, season, F, is_current)
            if ps is None:
                continue
            if gsis not in players_out:
                ref, sl = pl_idx.get(gsis, {}), sleeper.get(gsis, {})
                players_out[gsis] = {
                    "id": ref.get("sleeper_id") or sl.get("sleeper_id") or meta.get("sleeper_id") or gsis,
                    "gsis_id": gsis, "espn_id": ref.get("espn_id"), "yahoo_id": ref.get("yahoo_id"),
                    "name": ref.get("name") or sl.get("full_name") or meta["name"],
                    "pos": meta["pos"], "team": meta["team"],
                    "rookie": ref.get("rookie") or meta.get("rookie") or season,
                    "seasons": {},
                }
            if gsis in adp:
                ps["adp"] = adp[gsis]
            players_out[gsis]["seasons"][season] = ps
            players_out[gsis]["team"] = meta["team"] or players_out[gsis]["team"]

    print("Finalizing")
    for gsis, p in players_out.items():
        finalize_player(p, sleeper.get(gsis))
    result = [p for p in players_out.values() if len(p["seasons"]) >= args.min_seasons or args.end in p["seasons"]]
    result.sort(key=lambda p: (p["pos"], p["name"]))

    dataset = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "nflverse weekly_rosters + injuries + player_stats + nfldata schedules; Sleeper live status",
            "scoring": args.scoring, "currentSeason": args.end, "currentWeek": completed_current,
            "adp": None if args.no_adp else {"source": "MyFantasyLeague", "period": args.adp_period, "teams": args.adp_teams,
                                             "ppr": ppr_flag, "note": "overall rank across all positions, drafts before preseason"},
            "seasons": list(range(args.start, args.end + 1)), "players": len(result),
            "notes": "DNP = did not play without report entry; NR = not rostered; SUS = suspended/exempt. "
                     "NR, SUS, BYE, FUT are excluded from availability math.",
        },
        "players": result,
    }
    write_outputs(dataset, out)
    return dataset


def write_outputs(dataset, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset.json").write_text(json.dumps(dataset, separators=(",", ":"), default=_json_default))
    (out / "players").mkdir(exist_ok=True)
    rows = []
    for p in dataset["players"]:
        (out / "players" / f"{p['id']}.json").write_text(json.dumps(p, separators=(",", ":"), default=_json_default))
        for y, s in sorted(p["seasons"].items()):
            rows.append({"player_id": p["id"], "gsis_id": p.get("gsis_id"), "name": p["name"], "pos": p["pos"],
                         "team": p["team"], "season": y, "gp": s["gp"], "gm": s["gm"], "q": s["listed"]["Q"],
                         "d": s["listed"]["D"], "o": s["listed"]["O"], "ir": s["listed"]["IR"],
                         "injuries": s["injCount"], "ppg": s["ppg"], "pts_lost": s["ptsLost"],
                         "adp_pick": (s.get("adp") or {}).get("pick"), "adp_rank": (s.get("adp") or {}).get("rank"),
                         "adp_pos_rank": (s.get("adp") or {}).get("posRank"),
                         "ppg_estimated": s.get("ppgEstimated", False)})
    pd.DataFrame(rows).to_csv(out / "player_seasons.csv", index=False)
    print(f"Wrote {len(dataset['players'])} players, {len(rows)} player-seasons -> {out / 'dataset.json'}")


def _json_default(o):
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2026, help="current/latest season")
    ap.add_argument("--scoring", choices=["std", "half", "ppr"], default="half")
    ap.add_argument("--min-points", type=float, default=60.0,
                    help="season fantasy points needed to count as fantasy-relevant")
    ap.add_argument("--min-seasons", type=int, default=1)
    ap.add_argument("--sleeper-rank", type=int, default=250,
                    help="also include current-season players within this Sleeper search_rank")
    ap.add_argument("--out", default="./data")
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-sleeper", action="store_true")
    ap.add_argument("--no-adp", action="store_true", help="skip MFL ADP")
    ap.add_argument("--adp-period", default="JULY", choices=["JUNE", "JULY", "AUG1", "AUG15", "START", "ALL"],
                    help="MFL draft window; JULY = strictly before preseason game 1 (default)")
    ap.add_argument("--adp-teams", type=int, default=12, help="league size used for ADP (MFL FCOUNT)")
    ap.add_argument("--adp-keep", type=int, default=200,
                    help="always keep players drafted inside this overall ADP rank, even with 0 points")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
