# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

There is no build system, dependency manifest, linter, or test suite in this repo yet. Dependencies
(`nflreadpy`, `polars`, `requests`) must be installed manually into the active Python environment.

- Run the entry-point script: `python test.py` — this is a manual smoke test, not an automated test
  suite. It hits the live Sleeper API and downloads real NFL stats via `nflreadpy`, so it's slow and
  network-dependent, not something to run repeatedly in a tight loop.
- There is no `pytest`/`unittest` setup — if you add tests, you'll be introducing the test framework
  too, not just test files.

## Architecture

The pipeline computes **replacement-level fantasy points-per-game**, per position, for a specific
Sleeper league's actual scoring rules and roster requirements (see README.md for the "why"). Data
flows through four layers:

```
clients/  →  domain/  →  engine/  →  eval/
(raw API)   (typed models)  (stat→points)  (analysis)
```

- **`clients/`** — thin wrappers around external APIs; functions here return raw dicts/DataFrames
  in whatever shape the upstream source uses, with no domain logic.
  - `sleeper_client.py` — unauthenticated wrapper around Sleeper's public REST API
    (`get_user`, `get_user_leagues`, `get_league_users`, `get_league_rosters`).
  - `nflreadpy/player_data.py` — wraps the `nflreadpy` package. `load_player_metadata` joins
    `nfl.load_ff_playerids()` (which carries Sleeper IDs) against `nfl.load_players()` on `gsis_id`
    — this join is what bridges Sleeper's player IDs to `nflreadpy`'s `gsis_id`, which is the ID
    used everywhere downstream. `load_player_stats` joins raw per-game stats to that metadata and
    adds derived columns (`PROCESSED_STAT_COLUMNS`): threshold bonuses like `over_300_passing_yards`,
    position-gated `wr_receptions`/`rb_receptions`/`te_receptions`, combined `fumbles_lost`, and
    `opportunities` (targets + carries).
  - `nflreadpy/pbp_data.py` — wraps `nfl.load_pbp()` for raw play-by-play.

- **`domain/`** — dataclasses that model core concepts independent of any one API's raw shape, each
  with a `from_dict()` adapter that maps a specific upstream payload onto the model.
  - `scoring.py` — `ScoringSettings` is a flat dataclass with one `float` field per scoring rule
    Sleeper supports (passing/rushing/receiving/fumbles, all defaulting to `0.0`). `from_dict()`
    filters an arbitrary dict down to known field names, so it can be handed Sleeper's
    `scoring_settings` payload directly and silently drop kicker/defense/IDP keys it doesn't model.
  - `league.py` — `League` holds `league_id`, `league_name`, `season`, `scoring` (a `ScoringSettings`),
    `league_size`, and `starters` (a `dict[position, starter_count]`). `League.from_dict()` builds a
    `League` straight from a raw Sleeper league object (as returned by `get_user_leagues`):
    `total_rosters` → `league_size`, and `roster_positions` is tallied against `config.POSITIONS` to
    build `starters` — slots like `FLEX`/`BN`/`SUPER_FLEX` don't match any entry in `POSITIONS` and
    so aren't attributed to a single position.
  - `player.py` — `Player` wraps one player's metadata plus a small stat cache (`rolling_avg`,
    `boom_prob`, `bust_prob`) that `reset_cache()` recomputes from a stats DataFrame.

- **`engine/scoring.py`** — the translation layer between "how Sleeper names a scoring rule" and
  "how `nflreadpy` names the corresponding stat column." `SCORING_TO_STAT_COLUMN` maps
  `ScoringSettings` field names to `nflreadpy` column names; it is the single source of truth for
  which scoring rules are actually implemented. **Any `ScoringSettings` field absent from this dict
  is silently ignored** by `calculate_points_vectorized`, even if the league sets it nonzero (e.g.
  `pass_int_td`, `pass_td_40p`, `rec_0_4`/`rec_5_9`/etc. reception-distance buckets are defined on
  `ScoringSettings` but not currently wired into scoring). `calculate_points_vectorized` builds one
  polars expression summing `weight * stat_column` over every scoring field with a nonzero weight
  whose mapped column exists in the input DataFrame, and appends it as `fantasy_points`.

- **`eval/replacement_player.py`** — `get_replacement_level_ppgs(league, previous_season)` is the
  top-level analysis function. For each position in `config.POSITIONS`, it scores every player-game
  with `calculate_points_vectorized`, aggregates to points-per-game per player, drops players with
  fewer than `CUTOFF_GAME_AMT` games played, sorts descending by PPG, and reads off the player at
  rank `(league.league_size * league.starters_at(position)) + 1` — i.e. the best player who would
  *not* be a starter in a league of that size. That player's PPG is "replacement level" for the
  position.

`config.py` holds two shared constants used across layers: `PLAYER_METADATA` (the metadata columns
every player record carries) and `POSITIONS` (`['QB', 'RB', 'WR', 'TE']` — the only positions this
pipeline models; kicker/DEF/IDP are explicitly out of scope, per `domain/scoring.py`'s docstring).

`test.py` wires the whole pipeline together end-to-end (Sleeper account → league → `League.from_dict`
→ `get_replacement_level_ppgs`) and is currently the only executable entry point.
