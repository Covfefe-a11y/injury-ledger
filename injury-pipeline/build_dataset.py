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
  snap counts     https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.parquet
                  (offensive snap share per game -> detects players who left a game early)
  preseason ADP   https://fantasyfootballcalculator.com/api/v1/adp/{format}?teams=12&year={season}
                  format = standard | ppr | half-ppr (1QB, matched to --scoring) and 2qb (Superflex proxy).
                  FFC publishes the final preseason ADP window (roughly the last 10 days before kickoff)
                  for every year since 2010 and separates formats cleanly, which is why Josh Allen is
                  pick ~27 in 1QB half-PPR 2024 but 1.02 in 2QB.
                  Fallback: --adp-source mfl  (MyFantasyLeague JULY ADP, strictly before preseason game 1,
                  but ALL league formats mixed together, so QBs look far too early for 1QB leagues).
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
  on report (Q/D/O) & did not play  O     (missed, listed)  - the final game status is
                                           "out" whatever the Friday designation was;
                                           the designation is kept in `desig`
  report Q/D & played ............. Q     (played, listed)
  INA / no stats, no report ....... DNP   (NOT an injury: healthy scratch, benching,
                                           zero-stat game - excluded from all injury math)
  played, no report ............... H
  played but left early ........... H/Q + exit=true  (snap share < 50 % of his usual share, the drop is
                                           not a ramp-up, and he is on the injury report / out the
                                           following week; heuristic - see README)
"""
import argparse
import json
import math
import re
import statistics
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
    "stats_v2":  NFLVERSE + "/stats_player/stats_player_week_{season}.parquet",   # 2025+ location
    "players":   NFLVERSE + "/players/players.parquet",
    "schedule":  "https://github.com/nflverse/nfldata/raw/master/data/games.csv",
    "sleeper":   "https://api.sleeper.app/v1/players/nfl",
    "snaps":     NFLVERSE + "/snap_counts/snap_counts_{season}.parquet",
    "ffc":       "https://fantasyfootballcalculator.com/api/v1/adp/{fmt}?teams={teams}&year={season}&position=all",
    "adp":       "https://api.myfantasyleague.com/{season}/export?TYPE=adp&PERIOD={period}&FCOUNT={fcount}&IS_PPR={ppr}&IS_KEEPER=N&IS_MOCK=0&JSON=1",
    "playerids": "https://github.com/dynastyprocess/data/raw/master/files/db_playerids.csv",
}
# MFL ADP periods: JULY = drafts in July (strictly before any preseason game),
# AUG1 = first half of August (preseason week 1 usually falls in it), AUG15, START, ALL ...
FANTASY_POS = {"QB", "RB", "WR", "TE"}
FFC_FORMAT = {"half": "half-ppr", "ppr": "ppr", "std": "standard"}
# FFC spellings that differ from nflverse display names
NAME_ALIAS = {"hollywoodbrown": "marquisebrown", "robbiechosen": "robbyanderson", "robbieanderson": "robbyanderson",
              "mitchelltrubisky": "mitchtrubisky", "chigokonkwo": "chigoziemokonkwo", "kenwalker": "kennethwalker",
              "gabrieldavis": "gabedavis", "joshpalmer": "joshuapalmer", "scottymiller": "scottmiller",
              "nathanieldell": "tankdell", "demariodouglas": "popdouglas", "cjanderson": "cjanderson",
              "willfuller": "williamfuller", "dkmetcalf": "dkmetcalf", "ajdillon": "ajdillon"}
SUFFIXES = ("jr", "sr", "ii", "iii", "iv", "v")
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
        if "sleeper" in url:
            name = "sleeper_players.json"
        elif "fantasyfootballcalculator" in url:
            name = "ffc_" + re.sub(r"[^A-Za-z0-9]+", "_", url.split("/adp/")[1]) + ".json"
        else:
            name = "adp_" + url.split("/")[3] + "_" + url.split("PERIOD=")[1].split("&")[0] + ".json"
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
def name_key(name):
    """'Marvin Harrison Jr.' -> 'marvinharrison'; used to match ADP feeds to nflverse ids."""
    if not isinstance(name, str):
        return ""
    parts = re.sub(r"[^a-z ]", "", name.lower().replace("-", " ").replace(".", "")).split()
    while len(parts) > 1 and parts[-1] in SUFFIXES:
        parts.pop()
    key = "".join(parts)
    return NAME_ALIAS.get(key, key)


def season_frame(season, rosters, injuries, stats, schedule, scoring, snaps=None, pfr_to_gsis=None):
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

    # ----- snap counts: offensive snap share per game (for early-exit detection)
    snap_pct = {}
    if snaps is not None and len(snaps):
        sn = snaps.copy()
        sn = sn[first_col(sn, "game_type", "season_type", default="REG").fillna("REG").eq("REG")]
        sn["week"] = sn["week"].astype(int)
        pct = pd.to_numeric(first_col(sn, "offense_pct", default=None), errors="coerce")
        if pct is None or pct.isna().all():
            snaps_n = pd.to_numeric(first_col(sn, "offense_snaps", default=0), errors="coerce").fillna(0)
            team_max = snaps_n.groupby([sn["team"], sn["week"]]).transform("max")
            pct = snaps_n / team_max.replace(0, pd.NA)
        if pct.max() > 1.5:
            pct = pct / 100.0
        pfr = first_col(sn, "pfr_player_id", "pfr_id", default=None)
        names = first_col(sn, "player", "player_name", default=None)
        teams = first_col(sn, "team").map(norm_team)
        # name+team fallback index from this season's rosters
        by_name_team = defaultdict(set)
        for g, m in roster_meta.items():
            by_name_team[(name_key(m["name"]), m["team"])].add(g)
        for g, t in zip(ro["gsis_id"], ro["team_n"]):
            nm = roster_meta.get(g, {}).get("name")
            if nm:
                by_name_team[(name_key(nm), t)].add(g)
        for i, (w, v) in enumerate(zip(sn["week"], pct)):
            if pd.isna(v):
                continue
            g = None
            if pfr is not None and pfr_to_gsis:
                g = pfr_to_gsis.get(pfr.iloc[i])
            if g is None and names is not None:
                cands = by_name_team.get((name_key(names.iloc[i]), teams.iloc[i]), ())
                if len(cands) == 1:
                    g = next(iter(cands))
            if g is not None:
                snap_pct[(g, int(w))] = float(v)

    return {"n_weeks": n_weeks, "team_weeks": team_weeks, "completed": completed,
            "roster_idx": roster_idx, "roster_meta": roster_meta, "rep": rep, "played": played,
            "snaps": snap_pct}


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
        if rstat:
            row["desig"] = REPORT_TO_STATUS.get(rstat, rstat)   # Friday designation (Q/D/O), kept for reference

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
        else:  # did not play -> final game status is OUT if he was on the report at all
            if rstat in ("Out", "Doubtful", "Questionable") or body:
                row["status"], row["inj"] = "O", body
            else:
                row["status"] = "DNP"  # not injury related
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

    # ----- early exits: left the game hurt (snap share collapses, then on the report / out next week)
    sp = F.get("snaps") or {}
    played_rows = [r for r in weeks if r["played"]]
    for r in played_rows:
        v = sp.get((gsis, r["wk"]))
        if v is not None:
            r["snap"] = round(v, 2)
    shares = [(r["wk"], sp[(gsis, r["wk"])]) for r in played_rows if (gsis, r["wk"]) in sp]
    exits = 0
    if len(shares) >= 4:
        for idx, r in enumerate(weeks):
            if not r["played"] or (gsis, r["wk"]) not in sp:
                continue
            v = sp[(gsis, r["wk"])]
            others = [x for w2, x in shares if w2 != r["wk"]]
            if len(others) < 3:
                continue
            med = statistics.median(others)
            if med < 0.35 or v >= 0.5 * med:
                continue  # part-timer, or no real drop
            prev = [x for w2, x in shares if w2 < r["wk"]]
            if prev and prev[-1] < 0.5 * med:
                continue  # already low the week before -> ramp-up / role change, not an in-game exit
            nxt = next((weeks[j] for j in range(idx + 1, min(idx + 4, len(weeks)))
                        if weeks[j]["status"] not in ("BYE", "FUT")), None)
            if nxt is None:
                continue
            new_listing = nxt["status"] in ("Q", "D", "O", "IR") and (nxt.get("inj") or nxt.get("desig"))
            missed_next = nxt["status"] in ("O", "IR")
            if (r["status"] == "H" and new_listing) or (r["status"] == "Q" and missed_next):
                r["exit"] = True
                r["usual"] = round(med, 2)
                r["exitInj"] = nxt.get("inj")
                exits += 1
                for e in injuries:
                    if e["week"] == nxt["wk"]:
                        e["inGame"], e["exitWk"] = True, r["wk"]
                        break

    for e in injuries:
        e["sev"] = "major" if e["missed"] >= 4 else "mod" if e["missed"] >= 1 else "minor"
        del e["_end"]

    counted = [r for r in weeks if r["status"] not in ("BYE", "FUT", "NR", "SUS", "DNP")]
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
        "earlyExits": exits,
    }


# ------------------------------------------------ expected PPG from ADP ----
def _fit_loglin(pairs):
    """least squares y = a + b*ln(x) ; pairs = [(x, y)] -> (a, b, n)"""
    pts = [(math.log(x), y) for x, y in pairs if x and x > 0 and y is not None]
    n = len(pts)
    if n < 2:
        return None
    mx = sum(u for u, _ in pts) / n
    my = sum(v for _, v in pts) / n
    sxx = sum((u - mx) ** 2 for u, _ in pts)
    if sxx == 0:
        return None
    b = sum((u - mx) * (v - my) for u, v in pts) / sxx
    return (my - b * mx, b, n)


def _pos_rank_num(adp):
    pr = (adp or {}).get("posRank") or ""
    digits = "".join(ch for ch in pr if ch.isdigit())
    return int(digits) if digits else None


def fit_adp_curves(players_out, min_gp=8, min_n=15):
    """Per (season, pos) log-linear curve of realized PPG vs ADP position rank,
    pooled across seasons as fallback. Returns dict key -> (a, b, n)."""
    by_key = defaultdict(list)
    for p in players_out.values():
        for y, s in p["seasons"].items():
            r = _pos_rank_num(s.get("adp"))
            if r and s["ppg"] is not None and s["gp"] >= min_gp:
                by_key[(y, p["pos"])].append((r, s["ppg"]))
                by_key[("all", p["pos"])].append((r, s["ppg"]))
    curves = {}
    for key, pairs in by_key.items():
        fit = _fit_loglin(pairs)
        if fit and fit[1] < 0 and fit[2] >= (min_n if key[0] != "all" else 30):
            curves[key] = fit
    return curves


def apply_adp_expectation(players_out, curves, floor=2.0):
    n = 0
    for p in players_out.values():
        for y, s in p["seasons"].items():
            adp = s.get("adp")
            r = _pos_rank_num(adp)
            if not r:
                continue
            fit = curves.get((y, p["pos"])) or curves.get(("all", p["pos"]))
            if not fit:
                continue
            a, b, _ = fit
            exp_ppg = round(max(floor, a + b * math.log(r)), 2)
            adp["expPpg"] = exp_ppg
            adp["curve"] = "season" if (y, p["pos"]) in curves else "pooled"
            s["ptsLostAdp"] = round(s["gm"] * exp_ppg, 1)
            n += 1
            sf = s.get("adpSf")
            rsf = _pos_rank_num(sf)
            if sf and rsf:
                sf["expPpg"] = round(max(floor, a + b * math.log(rsf)), 2)
                sf["curve"] = adp["curve"]
                s["ptsLostAdpSf"] = round(s["gm"] * sf["expPpg"], 1)
    # Superflex-only entries (player drafted in 2QB but outside the 1QB list)
    for p in players_out.values():
        for y, s in p["seasons"].items():
            sf = s.get("adpSf")
            rsf = _pos_rank_num(sf)
            if sf and rsf and "expPpg" not in sf:
                fit = curves.get((y, p["pos"])) or curves.get(("all", p["pos"]))
                if fit:
                    sf["expPpg"] = round(max(floor, fit[0] + fit[1] * math.log(rsf)), 2)
                    sf["curve"] = "season" if (y, p["pos"]) in curves else "pooled"
                    s["ptsLostAdpSf"] = round(s["gm"] * sf["expPpg"], 1)
    return n


def finalize_player(p, sleeper):
    """Fill ppg gaps, points lost, current status."""
    seasons = p["seasons"]
    years = sorted(seasons)
    for i, y in enumerate(years):
        s = seasons[y]
        if s["ppg"] is None:
            cand = [seasons[z]["ppg"] for z in years[:i][::-1] + years[i + 1:] if seasons[z]["ppg"] is not None]
            if (s.get("adp") or {}).get("expPpg"):
                s["ppg"], s["ppgSource"] = s["adp"]["expPpg"], "adp"
            else:
                s["ppg"], s["ppgSource"] = (cand[0], "neighbour") if cand else (POS_BASE_PPG.get(p["pos"], 8.0), "position")
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
    adp_node = raw.get("adp") or {}
    players = adp_node.get("player") or (adp_node.get("players") or {}).get("player") or []
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


def load_adp_ffc(season, cache, refresh, fmt, teams, pl_idx, roster_meta):
    """FantasyFootballCalculator ADP for one season and format -> gsis_id -> adp dict.
    Players are matched by normalised name + position (+ team when ambiguous)."""
    url = URLS["ffc"].format(season=season, fmt=fmt, teams=teams)
    try:
        raw = fetch(url, cache, refresh, "json")
    except Exception as e:
        print(f"  ! FFC ADP unavailable for {season}/{fmt}: {e}")
        return {}, None
    if raw.get("status") != "Success" or not raw.get("players"):
        print(f"  ! FFC ADP empty for {season}/{fmt}: {str(raw)[:120]}")
        return {}, None
    meta = raw.get("meta") or {}
    # name index: season rosters first (right team), whole player index as fallback
    idx_season, idx_all = defaultdict(set), defaultdict(set)
    for g, m in roster_meta.items():
        if m["pos"] in FANTASY_POS:
            idx_season[(name_key(m["name"]), m["pos"])].add((g, m["team"]))
    for g, m in pl_idx.items():
        if m.get("pos") in FANTASY_POS:
            idx_all[(name_key(m.get("name")), m["pos"])].add((g, m.get("team")))
    rows = sorted(raw["players"], key=lambda r: float(r.get("adp") or 999))
    out, pos_counter, unmatched = {}, defaultdict(int), []
    for rank, r in enumerate(rows, start=1):
        pos = str(r.get("position") or "").upper()
        if pos not in FANTASY_POS:
            continue
        pos_counter[pos] += 1
        key = (name_key(r.get("name")), pos)
        team = norm_team(r.get("team"))
        cands = idx_season.get(key) or idx_all.get(key) or set()
        if len(cands) > 1:
            same_team = {c for c in cands if c[1] == team}
            cands = same_team or cands
        if len(cands) != 1:
            unmatched.append(f"{r.get('name')} ({pos}, {team})")
            continue
        gsis = next(iter(cands))[0]
        out[gsis] = {"pick": round(float(r["adp"]), 1), "rank": rank, "posRank": f"{pos}{pos_counter[pos]}",
                     "drafts": int(r.get("times_drafted") or 0), "source": "FFC", "format": meta.get("type") or fmt,
                     "high": r.get("high"), "low": r.get("low"), "stdev": r.get("stdev")}
    window = {"start": meta.get("start_date"), "end": meta.get("end_date"), "drafts": meta.get("total_drafts"),
              "type": meta.get("type"), "players": len(rows), "mapped": len(out)}
    print(f"  ADP {season}/{fmt}: {len(rows)} players, {len(out)} mapped, {meta.get('total_drafts')} drafts, "
          f"window {meta.get('start_date')}..{meta.get('end_date')}" + (f", unmatched: {', '.join(unmatched[:6])}" if unmatched else ""))
    return out, window


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
            "pfr_id": r.get("pfr_id") if isinstance(r.get("pfr_id"), str) else None,
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

    use_mfl = (not args.no_adp) and args.adp_source == "mfl"
    mfl_to_gsis = load_mfl_crosswalk(cache, args.refresh) if use_mfl else {}
    pos_of = {g: v.get("pos") for g, v in pl_idx.items()}
    ppr_flag = {"ppr": 1, "std": 0, "half": -1}[args.scoring]
    pfr_to_gsis = {v["pfr_id"]: g for g, v in pl_idx.items() if v.get("pfr_id")}
    adp_windows = {}
    players_out = {}
    completed_current = 0
    for season in range(args.start, args.end + 1):
        print(f"Season {season}")
        try:
            rosters = fetch(URLS["rosters"].format(season=season), cache, args.refresh, "parquet")
            injuries = fetch(URLS["injuries"].format(season=season), cache, args.refresh, "parquet")
            stats = None
            for key in ("stats_v2", "stats", "stats_new"):
                try:
                    stats = fetch(URLS[key].format(season=season), cache, args.refresh, "parquet")
                    break
                except FileNotFoundError:
                    continue
            if stats is None:
                raise FileNotFoundError(f"no weekly stats release found for {season}")
        except FileNotFoundError as e:
            print(f"  ! missing release asset, skipping season: {e}")
            continue
        snaps = None
        if not args.no_snaps:
            try:
                snaps = fetch(URLS["snaps"].format(season=season), cache, args.refresh, "parquet")
            except Exception as e:
                print(f"  ! snap counts unavailable for {season} (no early-exit detection): {e}")
        F = season_frame(season, rosters, injuries, stats, schedule, args.scoring, snaps, pfr_to_gsis)
        adp, adp_sf = {}, {}
        if not args.no_adp:
            if use_mfl:
                adp = load_adp(season, cache, args.refresh, args.adp_period, args.adp_teams, ppr_flag, mfl_to_gsis, pos_of)
            else:
                adp, w1 = load_adp_ffc(season, cache, args.refresh, FFC_FORMAT[args.scoring], args.adp_teams, pl_idx, F["roster_meta"])
                adp_sf, w2 = load_adp_ffc(season, cache, args.refresh, "2qb", args.adp_teams, pl_idx, F["roster_meta"])
                adp_windows[season] = {"1qb": w1, "sf": w2}
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
            if not relevant and ((gsis in adp and adp[gsis]["rank"] <= args.adp_keep) or
                                 (gsis in adp_sf and adp_sf[gsis]["rank"] <= args.adp_keep)):
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
            if gsis in adp_sf:
                ps["adpSf"] = adp_sf[gsis]
            players_out[gsis]["seasons"][season] = ps
            players_out[gsis]["team"] = meta["team"] or players_out[gsis]["team"]

    print("Finalizing")
    curves = fit_adp_curves(players_out)
    n_exp = apply_adp_expectation(players_out, curves)
    for pos in ("QB", "RB", "WR", "TE"):
        c = curves.get(("all", pos))
        if c:
            print(f"  ADP curve {pos} (pooled, n={c[2]}): PPG = {c[0]:.2f} {c[1]:+.2f}*ln(posRank)  ->  "
                  f"{pos}1 {c[0]:.1f}, {pos}12 {c[0]+c[1]*math.log(12):.1f}, {pos}24 {c[0]+c[1]*math.log(24):.1f}")
    print(f"  expected PPG attached to {n_exp} player-seasons with ADP")
    n_exits = sum(s.get("earlyExits", 0) for p in players_out.values() for s in p["seasons"].values())
    print(f"  early exits flagged: {n_exits}")
    for gsis, p in players_out.items():
        finalize_player(p, sleeper.get(gsis))
    result = [p for p in players_out.values() if len(p["seasons"]) >= args.min_seasons or args.end in p["seasons"]]
    result.sort(key=lambda p: (p["pos"], p["name"]))

    if args.no_adp:
        adp_meta = None
    elif use_mfl:
        adp_meta = {"source": "MyFantasyLeague", "period": args.adp_period, "teams": args.adp_teams, "ppr": ppr_flag,
                    "note": "overall rank across all positions, drafts before preseason; ALL league formats mixed (QBs inflated)"}
    else:
        adp_meta = {"source": "FantasyFootballCalculator", "teams": args.adp_teams,
                    "formats": {"1qb": FFC_FORMAT[args.scoring], "sf": "2qb"},
                    "note": "adp = 1QB (format matched to scoring), adpSf = 2QB/Superflex. Overall rank counts all positions incl. K/DST. "
                            "FFC's yearly ADP is the final preseason window (see windows), i.e. after the preseason games.",
                    "windows": adp_windows}
    if adp_meta is not None:
        adp_meta["expPpg"] = "per season+position log-linear fit of realized PPG (gp>=8) on 1QB ADP position rank; pooled fallback; applied to both formats"
        adp_meta["curves"] = {f"{k[0]}_{k[1]}": {"a": round(v[0], 3), "b": round(v[1], 3), "n": v[2]} for k, v in curves.items()}

    dataset = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "nflverse weekly_rosters + injuries + player_stats + nfldata schedules; Sleeper live status",
            "scoring": args.scoring, "currentSeason": args.end, "currentWeek": completed_current,
            "adp": adp_meta,
            "earlyExit": None if args.no_snaps else "week.exit=true when offensive snap share < 50% of the player's median share (>=3 other games, "
                                                    "median >= 35%), the previous game was not already low, and he is listed/out the next game",
            "seasons": list(range(args.start, args.end + 1)), "players": len(result),
            "notes": "DNP = did not play without any injury-report entry (healthy scratch / benching / zero-stat game) and is NOT "
                     "counted as an injury; NR = not rostered; SUS = suspended/exempt. DNP, NR, SUS, BYE, FUT are excluded from "
                     "availability math. Status O means the player did not play while on the report; the Friday designation is in desig.",
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
                         "adp_exp_ppg": (s.get("adp") or {}).get("expPpg"), "pts_lost_adp": s.get("ptsLostAdp"),
                         "adp_sf_pick": (s.get("adpSf") or {}).get("pick"), "adp_sf_rank": (s.get("adpSf") or {}).get("rank"),
                         "adp_sf_pos_rank": (s.get("adpSf") or {}).get("posRank"), "early_exits": s.get("earlyExits", 0),
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
    ap.add_argument("--no-adp", action="store_true", help="skip ADP entirely")
    ap.add_argument("--adp-source", choices=["ffc", "mfl"], default="ffc",
                    help="ffc = FantasyFootballCalculator (1QB + 2QB, clean formats, final preseason window; default). "
                         "mfl = MyFantasyLeague JULY ADP (pre-preseason, but all formats mixed)")
    ap.add_argument("--no-snaps", action="store_true", help="skip snap counts / early-exit detection")
    ap.add_argument("--adp-period", default="JULY", choices=["JUNE", "JULY", "AUG1", "AUG15", "START", "ALL"],
                    help="MFL draft window; JULY = strictly before preseason game 1 (default)")
    ap.add_argument("--adp-teams", type=int, default=12, help="league size used for ADP (FFC teams / MFL FCOUNT)")
    ap.add_argument("--adp-keep", type=int, default=200,
                    help="always keep players drafted inside this overall ADP rank, even with 0 points")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
