1
#!/usr/bin/env python3
2
"""
3
build_dataset.py  -  nflverse + Sleeper  ->  dataset.json for Injury Ledger
4


5
Sources (all free, no auth):
6
  weekly rosters  https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{season}.parquet
7
  injury reports  https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet
8
  player stats    https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{season}.parquet
9
                  (fallback: .../stats_player_week_{season}.parquet - newer naming)
10
  schedules       https://github.com/nflverse/nfldata/raw/master/data/games.csv
11
  player ids      https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet
12
  live status     https://api.sleeper.app/v1/players/nfl
13
  preseason ADP   https://api.myfantasyleague.com/{season}/export?TYPE=adp&PERIOD=JULY...
14
                  (drafts completed before preseason game 1; overall rank across all positions)
15
  id crosswalk    https://github.com/dynastyprocess/data/raw/master/files/db_playerids.csv (mfl_id -> gsis_id)
16


17
Output (default ./data):
18
  dataset.json              everything the HTML app needs (meta + players[])
19
  players/<sleeper_id>.json one file per player (for lazy loading later)
20
  player_seasons.csv        flat table for spreadsheets / BI
21


22
Usage:
23
  python build_dataset.py --start 2016 --end 2026 --scoring half --out ./data
24
  python build_dataset.py --no-sleeper          # skip live status
25
  python build_dataset.py --refresh             # ignore local cache
26


27
Weekly status mapping (per player, per regular-season week):
28
  team has no game ................ BYE   (excluded)
29
  week not yet played ............. FUT   (excluded)
30
  player not on any roster ........ NR    (excluded - unsigned / retired)
31
  roster status SUS/EXE ........... SUS   (excluded - not an injury)
32
  roster status RES/PUP/NON/IR .... IR    (missed, listed)
33
  report "Out" / "Doubtful" DNP ... O / D (missed, listed)
34
  report "Questionable" & played .. Q     (played, listed)
35
  report "Questionable" & DNP ..... O     (missed, listed)
36
  INA / no stats, no report ....... DNP   (missed, NOT listed - healthy scratch
37
                                            or zero-stat game)
38
  played, no report ............... H
39
"""
40
import argparse
41
import json
42
import math
43
import time
44
from collections import defaultdict
45
from pathlib import Path
46


47
import pandas as pd
48


49
try:
50
    import requests
51
except ImportError:  # pragma: no cover
52
    requests = None
53


54
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
55
URLS = {
56
    "rosters":   NFLVERSE + "/weekly_rosters/roster_weekly_{season}.parquet",
57
    "injuries":  NFLVERSE + "/injuries/injuries_{season}.parquet",
58
    "stats":     NFLVERSE + "/player_stats/player_stats_{season}.parquet",
59
    "stats_new": NFLVERSE + "/player_stats/stats_player_week_{season}.parquet",
60
    "stats_v2":  NFLVERSE + "/stats_player/stats_player_week_{season}.parquet",   # 2025+ location
61
    "players":   NFLVERSE + "/players/players.parquet",
62
    "schedule":  "https://github.com/nflverse/nfldata/raw/master/data/games.csv",
63
    "sleeper":   "https://api.sleeper.app/v1/players/nfl",
64
    "adp":       "https://api.myfantasyleague.com/{season}/export?TYPE=adp&PERIOD={period}&FCOUNT={fcount}&IS_PPR={ppr}&IS_KEEPER=N&IS_MOCK=0&JSON=1",
65
    "playerids": "https://github.com/dynastyprocess/data/raw/master/files/db_playerids.csv",
66
}
67
# MFL ADP periods: JULY = drafts in July (strictly before any preseason game),
68
# AUG1 = first half of August (preseason week 1 usually falls in it), AUG15, START, ALL ...
69
FANTASY_POS = {"QB", "RB", "WR", "TE"}
70
TEAM_FIX = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "JAC": "JAX", "WSH": "WAS"}
71
POS_BASE_PPG = {"QB": 17.0, "RB": 11.0, "WR": 11.0, "TE": 7.5}
72
RESERVE = ("RES", "PUP", "NON", "IR")
73


74
# body-part classification used by the app for soft-tissue vs structural views
75
SOFT = ("hamstring", "calf", "groin", "quad", "oblique", "thigh", "hip flexor", "soft")
76
HEAD = ("concussion", "head", "neck")
77
STRUCT = ("acl", "achilles", "lisfranc", "fracture", "broken", "collarbone", "clavicle",
78
          "meniscus", "back", "spine", "foot", "mcl", "pcl", "torn")
79


80


81
# ---------------------------------------------------------------- fetching ---
82
def fetch(url, cache, refresh, kind):
83
    cache.mkdir(parents=True, exist_ok=True)
84
    if kind == "json":
85
        name = "sleeper_players.json" if "sleeper" in url else "adp_" + url.split("/")[3] + "_" + url.split("PERIOD=")[1].split("&")[0] + ".json"
86
    else:
87
        name = url.split("/")[-1]
88
    local = cache / name
89
    fresh = local.exists() and not refresh
90
    if kind == "json" and "sleeper" in url and fresh and time.time() - local.stat().st_mtime > 3600:
91
        fresh = False  # live status: max 1h old
92
    if not fresh:
93
        if requests is None:
94
            raise SystemExit("python package 'requests' is missing")
95
        print(f"  GET {url}")
96
        r = requests.get(url, timeout=120)
97
        if r.status_code == 404:
98
            raise FileNotFoundError(url)
99
        r.raise_for_status()
100
        local.write_bytes(r.content)
101
    if kind == "parquet":
102
        return pd.read_parquet(local)
103
    if kind == "csv":
104
        return pd.read_csv(local, low_memory=False)
105
    return json.loads(local.read_text())
106


107


108
def is_na(v):
109
    return v is None or (isinstance(v, float) and math.isnan(v))
110


111


112
def norm_team(t):
113
    if is_na(t):
114
        return None
115
    t = str(t).upper()
116
    return TEAM_FIX.get(t, t)
117


118


119
def first_col(df, *names, default="__raise__"):
120
    for n in names:
121
        if n in df.columns:
122
            return df[n]
123
    if default == "__raise__":
124
        raise KeyError(f"none of {names} in columns {list(df.columns)[:25]}...")
125
    return pd.Series([default] * len(df), index=df.index, dtype="object")
126


127


128
def classify(body):
129
    if not body:
130
        return "unknown"
131
    b = body.lower()
132
    if any(k in b for k in HEAD):
133
        return "head"
134
    if any(k in b for k in SOFT):
135
        return "soft"
136
    if any(k in b for k in STRUCT):
137
        return "structural"
138
    return "joint"
139


140


141
def clean_id(v):
142
    if is_na(v):
143
        return None
144
    s = str(v)
145
    return s[:-2] if s.endswith(".0") else s
146


147


148
# ------------------------------------------------------------ transforms ----
149
def season_frame(season, rosters, injuries, stats, schedule, scoring):
150
    """Index everything for one season by (gsis_id, week)."""
151
    # ----- schedule: weeks, byes, completed weeks
152
    sch = schedule[(schedule["season"] == season) & (schedule["game_type"] == "REG")].copy()
153
    if sch.empty:
154
        raise ValueError(f"no schedule rows for {season}")
155
    n_weeks = int(sch["week"].max())
156
    team_weeks = defaultdict(set)
157
    for _, g in sch.iterrows():
158
        team_weeks[norm_team(g["home_team"])].add(int(g["week"]))
159
        team_weeks[norm_team(g["away_team"])].add(int(g["week"]))
160
    done = sch["result"].notna()
161
    completed = int(sch.loc[done, "week"].max()) if done.any() else 0
162


163
    # ----- weekly rosters: backbone
164
    ro = rosters.copy()
165
    ro = ro[first_col(ro, "game_type", "season_type", default="REG").fillna("REG").eq("REG")]
166
    ro["gsis_id"] = first_col(ro, "gsis_id", "player_id")
167
    ro["team_n"] = first_col(ro, "team").map(norm_team)
168
    ro["week"] = ro["week"].astype(int)
169
    ro["status_r"] = first_col(ro, "status", default="ACT").fillna("ACT").astype(str).str.upper()
170
    ro = ro[ro["gsis_id"].notna()]
171
    roster_idx = {(g, w): (t, s) for g, w, t, s in zip(ro["gsis_id"], ro["week"], ro["team_n"], ro["status_r"])}
172
    roster_meta = {}
173
    name_col = first_col(ro, "full_name", "player_name", default="")
174
    sl_col = first_col(ro, "sleeper_id", default=None)
175
    rk_col = first_col(ro, "rookie_year", "rookie_season", "entry_year", default=None)
176
    ro = ro.assign(_name=name_col, _sl=sl_col, _rk=rk_col).sort_values("week")
177
    for g, grp in ro.groupby("gsis_id"):
178
        last = grp.iloc[-1]
179
        sl = grp["_sl"].dropna()
180
        rk = grp["_rk"].dropna()
181
        roster_meta[g] = {
182
            "name": str(last["_name"] or ""),
183
            "pos": str(first_col(grp, "position", default="").iloc[-1] or ""),
184
            "team": last["team_n"],
185
            "sleeper_id": clean_id(sl.iloc[-1]) if len(sl) else None,
186
            "rookie": int(rk.iloc[0]) if len(rk) else None,
187
        }
188


189
    # ----- injury reports
190
    inj = injuries.copy()
191
    inj = inj[first_col(inj, "game_type", "season_type", default="REG").fillna("REG").eq("REG")]
192
    inj["week"] = inj["week"].astype(int)
193
    rep = {}
194
    for r in inj.to_dict("records"):
195
        status = r.get("report_status")
196
        body = r.get("report_primary_injury")
197
        if not isinstance(body, str):
198
            body = r.get("practice_primary_injury")
199
        if not isinstance(status, str) and not isinstance(body, str):
200
            continue
201
        rep[(r["gsis_id"], int(r["week"]))] = {
202
            "status": status.strip().title() if isinstance(status, str) else None,
203
            "body": body.strip().title() if isinstance(body, str) else None,
204
        }
205


206
    # ----- stats: played + points
207
    st = stats.copy()
208
    st = st[first_col(st, "season_type", "game_type", default="REG").fillna("REG").eq("REG")]
209
    st["gsis_id"] = first_col(st, "player_id", "gsis_id")
210
    st["week"] = st["week"].astype(int)
211
    std = pd.to_numeric(first_col(st, "fantasy_points", default=0.0), errors="coerce").fillna(0.0)
212
    ppr = pd.to_numeric(first_col(st, "fantasy_points_ppr", default=None), errors="coerce")
213
    ppr = ppr.fillna(std) if ppr.notna().any() else std
214
    pts = {"std": std, "ppr": ppr, "half": (std + ppr) / 2}[scoring]
215
    played = {(g, w): float(p) for g, w, p in zip(st["gsis_id"], st["week"], pts)}
216


217
    return {"n_weeks": n_weeks, "team_weeks": team_weeks, "completed": completed,
218
            "roster_idx": roster_idx, "roster_meta": roster_meta, "rep": rep, "played": played}
219


220


221
REPORT_TO_STATUS = {"Out": "O", "Doubtful": "D", "Questionable": "Q"}
222


223


224
def build_player_season(gsis, season, F, is_current):
225
    n_weeks, completed = F["n_weeks"], F["completed"]
226
    weeks, injuries = [], []
227
    team_prev, ever_on_roster = None, False
228
    for w in range(1, n_weeks + 1):
229
        key = (gsis, w)
230
        roster = F["roster_idx"].get(key)
231
        team = roster[0] if roster else team_prev
232
        rstatus = roster[1] if roster else None
233
        rep = F["rep"].get(key)
234
        pts = F["played"].get(key)
235
        body = rep["body"] if rep else None
236
        rstat = rep["status"] if rep else None
237
        row = {"wk": w, "status": "H", "played": False, "inj": None}
238


239
        if is_current and w > completed:
240
            row["status"] = "FUT"
241
            if w == completed + 1 and (rep or rstatus in RESERVE):
242
                nxt = "IR" if rstatus in RESERVE else REPORT_TO_STATUS.get(rstat, "Q")
243
                row["next"] = {"status": nxt, "inj": body}
244
        elif roster is None and pts is None and rep is None:
245
            row["status"] = "NR"
246
        elif team and w not in F["team_weeks"].get(team, set()):
247
            row["status"] = "BYE"
248
        elif rstatus in ("SUS", "EXE"):
249
            row["status"] = "SUS"
250
        elif rstatus in RESERVE and pts is None:
251
            row["status"] = "IR"
252
            row["inj"] = body or (injuries[-1]["type"] if injuries else "Reserve list")
253
        elif pts is not None:
254
            row["played"] = True
255
            if rstat in ("Questionable", "Doubtful"):
256
                row["status"], row["inj"] = "Q", body
257
        else:  # did not play
258
            if rstat == "Doubtful":
259
                row["status"], row["inj"] = "D", body
260
            elif rstat in ("Out", "Questionable") or body:
261
                row["status"], row["inj"] = "O", body
262
            else:
263
                row["status"] = "DNP"
264
        if roster:
265
            ever_on_roster = True
266
        team_prev = team
267
        weeks.append(row)
268


269
        # injury events: consecutive listed weeks with the same body part
270
        if row["status"] in ("Q", "D", "O", "IR") and row["inj"]:
271
            last = injuries[-1] if injuries else None
272
            missed = 0 if row["played"] else 1
273
            if last and last["_end"] == w - 1 and last["type"] == row["inj"]:
274
                last["_end"] = w
275
                last["listed"] += 1
276
                last["missed"] += missed
277
            else:
278
                injuries.append({"week": w, "type": row["inj"], "cls": classify(row["inj"]),
279
                                 "sev": None, "missed": missed, "listed": 1, "_end": w})
280


281
    if not ever_on_roster and not any(r["played"] for r in weeks):
282
        return None  # no NFL season for this player
283


284
    for e in injuries:
285
        e["sev"] = "major" if e["missed"] >= 4 else "mod" if e["missed"] >= 1 else "minor"
286
        del e["_end"]
287


288
    counted = [r for r in weeks if r["status"] not in ("BYE", "FUT", "NR", "SUS")]
289
    gp = sum(1 for r in counted if r["played"])
290
    gm = len(counted) - gp
291
    pts_played = [F["played"][(gsis, r["wk"])] for r in weeks if r["played"]]
292
    ppg = round(sum(pts_played) / len(pts_played), 2) if pts_played else None
293
    listed = {k: sum(1 for r in counted if r["status"] == k) for k in ("Q", "D", "O", "IR")}
294
    return {
295
        "season": season, "nWeeks": n_weeks,
296
        "bye": next((r["wk"] for r in weeks if r["status"] == "BYE"), None),
297
        "ppg": ppg, "weeks": weeks, "injuries": injuries,
298
        "gp": gp, "gm": gm, "gamesPossible": len(counted), "listed": listed,
299
        "listedTotal": sum(listed.values()),
300
        "listedWeighted": listed["Q"] * 0.25 + listed["D"] * 0.5 + listed["O"] + listed["IR"],
301
        "injCount": len(injuries), "hadIR": listed["IR"] > 0,
302
        "ptsLost": round(gm * ppg, 1) if ppg is not None else None,
303
    }
304


305


306
def finalize_player(p, sleeper):
307
    """Fill ppg gaps, points lost, current status."""
308
    seasons = p["seasons"]
309
    years = sorted(seasons)
310
    for i, y in enumerate(years):
311
        s = seasons[y]
312
        if s["ppg"] is None:
313
            cand = [seasons[z]["ppg"] for z in years[:i][::-1] + years[i + 1:] if seasons[z]["ppg"] is not None]
314
            s["ppg"] = cand[0] if cand else POS_BASE_PPG.get(p["pos"], 8.0)
315
            s["ppgEstimated"] = True
316
            s["ptsLost"] = round(s["gm"] * s["ppg"], 1)
317
    cur = {"status": "H", "inj": None}
318
    if sleeper:
319
        st = (sleeper.get("injury_status") or "").upper()
320
        m = {"IR": "IR", "OUT": "O", "DOUBTFUL": "D", "QUESTIONABLE": "Q", "PUP": "IR", "NA": "IR", "COV": "O"}
321
        if st in m:
322
            cur = {"status": m[st], "inj": sleeper.get("injury_body_part") or sleeper.get("injury_notes"),
323
                   "since": sleeper.get("injury_start_date")}
324
        if sleeper.get("team"):
325
            p["team"] = norm_team(sleeper["team"])
326
    elif years:
327
        last = seasons[years[-1]]
328
        nxt = next((r.get("next") for r in last["weeks"] if r.get("next")), None)
329
        if nxt:
330
            cur = nxt
331
    p["current"] = cur
332


333


334
# -------------------------------------------------------------------- ADP ----
335
def load_adp(season, cache, refresh, period, fcount, ppr_flag, mfl_to_gsis, pos_of):
336
    """Return gsis_id -> {pick, rank, posRank, drafts, period} from MFL preseason ADP."""
337
    url = URLS["adp"].format(season=season, period=period, fcount=fcount, ppr=ppr_flag)
338
    try:
339
        raw = fetch(url, cache, refresh, "json")
340
    except Exception as e:
341
        print(f"  ! ADP unavailable for {season}: {e}")
342
        return {}
343
    adp_node = raw.get("adp") or {}
344
    players = adp_node.get("player") or (adp_node.get("players") or {}).get("player") or []
345
    if isinstance(players, dict):
346
        players = [players]
347
    rows = []
348
    for r in players:
349
        try:
350
            rows.append((float(r["averagePick"]), str(r["id"]), int(float(r.get("draftsSelectedIn", 0) or 0))))
351
        except (KeyError, ValueError, TypeError):
352
            continue
353
    rows.sort()
354
    out, pos_counter = {}, defaultdict(int)
355
    rank = 0
356
    for pick, mfl_id, drafts in rows:
357
        rank += 1
358
        gsis = mfl_to_gsis.get(mfl_id)
359
        pos = pos_of.get(gsis) if gsis else None
360
        if pos:
361
            pos_counter[pos] += 1
362
        if gsis:
363
            out[gsis] = {"pick": round(pick, 1), "rank": rank, "posRank": f"{pos}{pos_counter[pos]}" if pos else None,
364
                         "drafts": drafts, "period": period, "source": "MFL"}
365
    print(f"  ADP {season}/{period}: {len(rows)} players, {len(out)} mapped to gsis ids, {raw.get('adp', {}).get('totalDrafts', '?')} drafts")
366
    return out
367


368


369
def load_mfl_crosswalk(cache, refresh):
370
    try:
371
        df = fetch(URLS["playerids"], cache, refresh, "csv")
372
    except Exception as e:
373
        print(f"  ! player id crosswalk unavailable: {e}")
374
        return {}
375
    df = df[df["mfl_id"].notna() & df["gsis_id"].notna()]
376
    return {clean_id(m): str(g) for m, g in zip(df["mfl_id"], df["gsis_id"])}
377


378


379
# ------------------------------------------------------------------ main ----
380
def load_player_index(players):
381
    players = players.copy()
382
    players["gsis_id"] = first_col(players, "gsis_id")
383
    rk = "rookie_season" if "rookie_season" in players.columns else "rookie_year"
384
    idx = {}
385
    for r in players[players["gsis_id"].notna()].to_dict("records"):
386
        idx[r["gsis_id"]] = {
387
            "name": r.get("display_name") or r.get("full_name"),
388
            "pos": r.get("position"),
389
            "team": norm_team(r.get("latest_team") or r.get("team_abbr") or r.get("team")),
390
            "sleeper_id": clean_id(r.get("sleeper_id")),
391
            "espn_id": clean_id(r.get("espn_id")), "yahoo_id": clean_id(r.get("yahoo_id")),
392
            "rookie": None if is_na(r.get(rk)) else int(r.get(rk)),
393
        }
394
    return idx
395


396


397
def run(args):
398
    cache, out = Path(args.cache), Path(args.out)
399
    print("Loading reference data")
400
    schedule = fetch(URLS["schedule"], cache, args.refresh, "csv")
401
    pl_idx = load_player_index(fetch(URLS["players"], cache, args.refresh, "parquet"))
402
    sleeper = {}
403
    if not args.no_sleeper:
404
        try:
405
            raw = fetch(URLS["sleeper"], cache, args.refresh, "json")
406
            sleeper = {v.get("gsis_id"): dict(v, sleeper_id=k) for k, v in raw.items() if v.get("gsis_id")}
407
            print(f"  Sleeper: {len(raw)} players, {len(sleeper)} with gsis_id")
408
        except Exception as e:  # live status is optional
409
            print(f"  ! Sleeper status unavailable: {e}")
410


411
    mfl_to_gsis = {} if args.no_adp else load_mfl_crosswalk(cache, args.refresh)
412
    pos_of = {g: v.get("pos") for g, v in pl_idx.items()}
413
    ppr_flag = {"ppr": 1, "std": 0, "half": -1}[args.scoring]
414
    players_out = {}
415
    completed_current = 0
416
    for season in range(args.start, args.end + 1):
417
        print(f"Season {season}")
418
        try:
419
            rosters = fetch(URLS["rosters"].format(season=season), cache, args.refresh, "parquet")
420
            injuries = fetch(URLS["injuries"].format(season=season), cache, args.refresh, "parquet")
421
            stats = None
422
            for key in ("stats_v2", "stats", "stats_new"):
423
                try:
424
                    stats = fetch(URLS[key].format(season=season), cache, args.refresh, "parquet")
425
                    break
426
                except FileNotFoundError:
427
                    continue
428
            if stats is None:
429
                raise FileNotFoundError(f"no weekly stats release found for {season}")
430
        except FileNotFoundError as e:
431
            print(f"  ! missing release asset, skipping season: {e}")
432
            continue
433
        F = season_frame(season, rosters, injuries, stats, schedule, args.scoring)
434
        adp = {} if args.no_adp else load_adp(season, cache, args.refresh, args.adp_period, args.adp_teams, ppr_flag, mfl_to_gsis, pos_of)
435
        is_current = season == args.end
436
        if is_current:
437
            completed_current = F["completed"]
438
        season_pts = defaultdict(float)
439
        for (g, _w), p in F["played"].items():
440
            season_pts[g] += p
441
        for gsis, meta in F["roster_meta"].items():
442
            if meta["pos"] not in FANTASY_POS:
443
                continue
444
            relevant = season_pts.get(gsis, 0.0) >= args.min_points or gsis in players_out
445
            if not relevant and gsis in adp and adp[gsis]["rank"] <= args.adp_keep:
446
                relevant = True  # drafted early but scored nothing (e.g. preseason injury) - exactly what we track
447
            if not relevant and is_current:
448
                relevant = (sleeper.get(gsis, {}).get("search_rank") or 9999) <= args.sleeper_rank
449
            if not relevant:
450
                continue
451
            ps = build_player_season(gsis, season, F, is_current)
452
            if ps is None:
453
                continue
454
            if gsis not in players_out:
455
                ref, sl = pl_idx.get(gsis, {}), sleeper.get(gsis, {})
456
                players_out[gsis] = {
457
                    "id": ref.get("sleeper_id") or sl.get("sleeper_id") or meta.get("sleeper_id") or gsis,
458
                    "gsis_id": gsis, "espn_id": ref.get("espn_id"), "yahoo_id": ref.get("yahoo_id"),
459
                    "name": ref.get("name") or sl.get("full_name") or meta["name"],
460
                    "pos": meta["pos"], "team": meta["team"],
461
                    "rookie": ref.get("rookie") or meta.get("rookie") or season,
462
                    "seasons": {},
463
                }
464
            if gsis in adp:
465
                ps["adp"] = adp[gsis]
466
            players_out[gsis]["seasons"][season] = ps
467
            players_out[gsis]["team"] = meta["team"] or players_out[gsis]["team"]
468


469
    print("Finalizing")
470
    for gsis, p in players_out.items():
471
        finalize_player(p, sleeper.get(gsis))
472
    result = [p for p in players_out.values() if len(p["seasons"]) >= args.min_seasons or args.end in p["seasons"]]
473
    result.sort(key=lambda p: (p["pos"], p["name"]))
474


475
    dataset = {
476
        "meta": {
477
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
478
            "source": "nflverse weekly_rosters + injuries + player_stats + nfldata schedules; Sleeper live status",
479
            "scoring": args.scoring, "currentSeason": args.end, "currentWeek": completed_current,
480
            "adp": None if args.no_adp else {"source": "MyFantasyLeague", "period": args.adp_period, "teams": args.adp_teams,
481
                                             "ppr": ppr_flag, "note": "overall rank across all positions, drafts before preseason"},
482
            "seasons": list(range(args.start, args.end + 1)), "players": len(result),
483
            "notes": "DNP = did not play without report entry; NR = not rostered; SUS = suspended/exempt. "
484
                     "NR, SUS, BYE, FUT are excluded from availability math.",
485
        },
486
        "players": result,
487
    }
488
    write_outputs(dataset, out)
489
    return dataset
490


491


492
def write_outputs(dataset, out):
493
    out.mkdir(parents=True, exist_ok=True)
494
    (out / "dataset.json").write_text(json.dumps(dataset, separators=(",", ":"), default=_json_default))
495
    (out / "players").mkdir(exist_ok=True)
496
    rows = []
497
    for p in dataset["players"]:
498
        (out / "players" / f"{p['id']}.json").write_text(json.dumps(p, separators=(",", ":"), default=_json_default))
499
        for y, s in sorted(p["seasons"].items()):
500
            rows.append({"player_id": p["id"], "gsis_id": p.get("gsis_id"), "name": p["name"], "pos": p["pos"],
501
                         "team": p["team"], "season": y, "gp": s["gp"], "gm": s["gm"], "q": s["listed"]["Q"],
502
                         "d": s["listed"]["D"], "o": s["listed"]["O"], "ir": s["listed"]["IR"],
503
                         "injuries": s["injCount"], "ppg": s["ppg"], "pts_lost": s["ptsLost"],
504
                         "adp_pick": (s.get("adp") or {}).get("pick"), "adp_rank": (s.get("adp") or {}).get("rank"),
505
                         "adp_pos_rank": (s.get("adp") or {}).get("posRank"),
506
                         "ppg_estimated": s.get("ppgEstimated", False)})
507
    pd.DataFrame(rows).to_csv(out / "player_seasons.csv", index=False)
508
    print(f"Wrote {len(dataset['players'])} players, {len(rows)} player-seasons -> {out / 'dataset.json'}")
509


510


511
def _json_default(o):
512
    if isinstance(o, pd.Timestamp):
513
        return o.isoformat()
514
    if hasattr(o, "item"):
515
        return o.item()
516
    if isinstance(o, float) and math.isnan(o):
517
        return None
518
    return str(o)
519


520


521
def parse_args(argv=None):
522
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
523
    ap.add_argument("--start", type=int, default=2016)
524
    ap.add_argument("--end", type=int, default=2026, help="current/latest season")
525
    ap.add_argument("--scoring", choices=["std", "half", "ppr"], default="half")
526
    ap.add_argument("--min-points", type=float, default=60.0,
527
                    help="season fantasy points needed to count as fantasy-relevant")
528
    ap.add_argument("--min-seasons", type=int, default=1)
529
    ap.add_argument("--sleeper-rank", type=int, default=250,
530
                    help="also include current-season players within this Sleeper search_rank")
531
    ap.add_argument("--out", default="./data")
532
    ap.add_argument("--cache", default="./cache")
533
    ap.add_argument("--refresh", action="store_true")
534
    ap.add_argument("--no-sleeper", action="store_true")
535
    ap.add_argument("--no-adp", action="store_true", help="skip MFL ADP")
536
    ap.add_argument("--adp-period", default="JULY", choices=["JUNE", "JULY", "AUG1", "AUG15", "START", "ALL"],
537
                    help="MFL draft window; JULY = strictly before preseason game 1 (default)")
538
    ap.add_argument("--adp-teams", type=int, default=12, help="league size used for ADP (MFL FCOUNT)")
539
    ap.add_argument("--adp-keep", type=int, default=200,
540
                    help="always keep players drafted inside this overall ADP rank, even with 0 points")
541
    return ap.parse_args(argv)
542


543


544
if __name__ == "__main__":
545
    run(parse_args())
546

