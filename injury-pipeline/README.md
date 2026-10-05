# Injury Ledger data pipeline (nflverse → dataset.json)

Backfills 10+ seasons of weekly NFL injury data for fantasy-relevant players and
emits exactly the JSON that `fantasy-injury-tracker.html` consumes.

## Quick start

```bash
python -m pip install -r requirements.txt
python build_dataset.py --start 2016 --end 2026 --scoring half --out ./data
# first run downloads ~300 MB of parquet into ./cache; later runs only refetch what changed
```

Serve the app and the data from the same folder:

```bash
cp ../fantasy-injury-tracker.html ./data/index.html
cd data && python -m http.server 8000      # open http://localhost:8000
```

The app looks for `dataset.json` next to itself, or at `?data=https://…/dataset.json`.
Browsers block `fetch()` from `file://`, so double-clicking the HTML keeps the
synthetic sample data — run a local server or host it (GitHub Pages works).

## Sources (all free, no API key)

| What | Where | Since |
|---|---|---|
| Weekly roster status (ACT / RES=IR / PUP / INA / SUS …) | nflverse `weekly_rosters` | 2002 |
| Official injury report (status + body part) | nflverse `injuries` | 2009 |
| Weekly stats + fantasy points (std / PPR) | nflverse `player_stats` | 1999 |
| Schedule (byes, completed weeks) | nfldata `games.csv` | 1999 |
| ID crosswalk (gsis → sleeper / espn / yahoo) | nflverse `players` | – |
| Live designation for the "Now" column | Sleeper `/v1/players/nfl` | live |
| Preseason ADP (Top 12 / 24 / 50 / 100 filter) | MyFantasyLeague `export?TYPE=adp` + DynastyProcess `db_playerids.csv` (mfl_id → gsis_id) | 2007 |

## Weekly status rules

| Situation | Status | Counted as |
|---|---|---|
| Team has no game | `BYE` | excluded |
| Week not yet played (current season) | `FUT` | excluded |
| Player on no roster | `NR` | excluded |
| Roster status SUS / EXE | `SUS` | excluded (not an injury) |
| Roster status RES / PUP / NON | `IR` | missed + listed |
| Report Out / Doubtful, did not play | `O` / `D` | missed + listed |
| Report Questionable, played | `Q` | played + listed |
| Report Questionable, did not play | `O` | missed + listed |
| No stats, no report (healthy scratch / zero-stat game) | `DNP` | missed, **not** listed |
| Played, no report | `H` | played |

Injury *events* = consecutive listed weeks with the same body part.
Severity: `major` ≥ 4 games missed, `mod` ≥ 1, `minor` = played through it.
Points lost = games missed × that season's PPG (neighbouring season's PPG if the
player never played, flagged `ppgEstimated`).

## Preseason ADP

Each player-season carries `adp: {pick, rank, posRank, drafts, period, source}` taken
from MyFantasyLeague's public ADP export. `rank` is the **overall** rank across all
positions (what "Top 12 / 24 / 50 / 100" means in the app). The default window is
`--adp-period JULY` = real (non-mock, non-keeper) drafts completed in July, i.e.
strictly before preseason game 1. `AUG1` gives a larger sample but overlaps the
first preseason weekend. `--adp-teams 12` sets the league size; `--adp-keep 200`
forces players drafted inside the top 200 into the dataset even if they scored
zero points that year (camp injuries are exactly what you want to see).

## Options

```
--scoring std|half|ppr   fantasy points basis (default half)
--min-points 60          season points needed to be "fantasy relevant"; once a player
                         qualifies, all later seasons are kept
--sleeper-rank 250       also keep current-season players inside Sleeper's top-N search rank
--min-seasons 1          drop players with fewer seasons (unless active now)
--no-sleeper             skip the live status call
--no-adp                 skip MFL ADP
--adp-period JULY        JUNE|JULY|AUG1|AUG15|START|ALL
--adp-teams 12           league size for ADP
--adp-keep 200           keep anyone drafted inside this overall ADP rank
--refresh                ignore ./cache
```

## Output

```
data/dataset.json          { meta:{currentSeason, currentWeek, seasons, scoring, …}, players:[…] }
data/players/<id>.json     one file per player, keyed by Sleeper ID (gsis_id if unknown)
data/player_seasons.csv    flat table, one row per player-season (Excel / BI)
```

Player object (what the HTML app reads):

```
{ id, gsis_id, espn_id, yahoo_id, name, pos, team, rookie, current:{status,inj,since},
  seasons:{ 2024:{ season, nWeeks, bye, ppg, weeks:[{wk,status,played,inj,next?}],
                   injuries:[{week,type,cls,sev,missed,listed}], adp:{pick,rank,posRank,drafts,period},
                   gp, gm, listed:{Q,D,O,IR}, ptsLost, … } } }
```

## Automation

`.github/workflows/build-dataset.yml` rebuilds daily (plus every 3 h Thu–Sun) and
commits `data/`. Enable GitHub Pages on the repo and open
`https://<user>.github.io/<repo>/data/` with the HTML copied in as `index.html`.

## Known limitations

* nflverse `injuries` lists only players on the official report; IR stints are
  detected from weekly roster status instead. Body part for IR weeks is inherited
  from the last report entry (or "Reserve list" if there was none, e.g. preseason ACL).
* "Played" = appears in weekly stats. A player who was active but recorded zero
  touches/targets looks like `DNP`. Join nflverse `snap_counts` (PFR IDs) for
  snap-level precision.
* Column names in nflverse releases change occasionally; `first_col()` already
  handles the known variants (`recent_team`/`team`, `rookie_year`/`rookie_season`,
  `player_stats_*`/`stats_player_week_*`).
* ESPN / Yahoo IDs are included in the output so a later OAuth roster import can
  match players directly.
