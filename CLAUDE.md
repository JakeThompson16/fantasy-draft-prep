# fantasy-draft-prep — CLAUDE.md

## What this is
A pre-draft cheat sheet generator. Pulls a specific Sleeper league's real
scoring settings and roster requirements, scores last season's NFL stats
under those exact settings, computes replacement level per position, and
ranks every player by value-over-replacement (VOR). Output is a sortable
board to reference live during tonight's draft.

**Time-boxed. Draft is tonight. Ship Phase 1 completely before touching
Phase 2. Phase 2 is explicitly optional — if it's not done cleanly with
time to spare, cut it and ship Phase 1 alone.**

## Explicitly out of scope tonight — do not build these
- Live draft pick tracking / polling `/draft/{id}/picks`. Static board only.
- FLEX-slot-aware replacement level. Replacement level uses **dedicated
  starter slots only** (`League.starters`, built from `roster_positions`
  matches to QB/RB/WR/TE — FLEX/BN are deliberately not tallied). This
  was a conscious simplification, not an oversight — don't "fix" it.
- Multi-year weighted average. Single prior season PPG only.
- ADP / consensus rank integration. No market-comparison layer tonight.
- Any change to `domain/player.py`. Its `rolling_avg`/`boom_prob`/
  `bust_prob` fields are from an earlier, different design (in-season
  rolling projection) and are **not used anywhere in this pipeline**.
  Don't wire it in, don't "clean it up" — leave it alone or ignore it.
- Any of the `ScoringSettings` fields with no entry in
  `SCORING_TO_STAT_COLUMN` (big-play TD bonuses, reception-yardage
  buckets, first-down bonuses beyond what's mapped). Confirmed unused by
  this league. Don't add play-by-play aggregation to support them.

## Architecture
- `domain/scoring.py` — `ScoringSettings` dataclass. `.from_dict()` builds
  it from Sleeper's raw `scoring_settings` payload, silently dropping any
  key with no matching field (intentional — IDP/kicker/defense are out of
  scope).
- `domain/league.py` — `League` dataclass. `.from_dict()` builds it from a
  raw Sleeper league payload. `.starters_at(position)` returns dedicated
  starter slot count. `.league_size` is `total_rosters`.
- `engine/scoring.py` — `calculate_points_vectorized(stats_df, scoring)`
  adds a `fantasy_points` column via `SCORING_TO_STAT_COLUMN` mapping.
  Vectorized polars expression, one term per nonzero scoring weight.
- `clients/sleeper_client/` — thin REST wrapper, no auth.
  `get_user(username)` → `get_user_leagues(user_id, season)` is the
  required two-step (there's no username-keyed leagues endpoint — this
  bit us once already tonight, don't repeat it).
- `clients/nflreadpy/player_data.py` — `load_player_stats(season)` returns
  one row per player per game with `gsis_id`, `position`, `display_name`,
  and every column `engine/scoring.py` needs. Already joined/deduped.
- `eval/replacement_player.py` — replacement level calculator. **Refactor
  target for Phase 1** (see Task 1 below).
- `model/features.py` — `build_features(stats_df, snap_df, season)`, the
  one shared feature function for both historical training rows and live
  scoring. `previous_season_vor` is NOT produced here (it needs a
  `League`); callers join `get_player_vor(league, season)`'s `vor` column
  on afterward, keyed on `gsis_id` + `position`.
- `model/historical_vor.py` — `build_season_transitions(league)` applies
  this league's current scoring/roster rules retroactively to every
  season 2019-2025 and pairs each season N with N+1, filtered to
  `TRAINING_GAMES_FLOOR` (6 games in both seasons). `add_boom_label` /
  `print_training_row_counts` / `print_class_balance` are the
  checkpoints run before any modeling.
- `model/train.py` — `build_training_table(league)` joins
  `build_features()` onto the transitions, drops rows with any null
  feature. `train_position_models(league)` fits 4 independent
  `LogisticRegression` (default L2), each on features scaled by a
  `StandardScaler` fit on that position's training rows only, and prints
  Brier score + log loss on the held-out 2024→2025 transition.

## Data conventions
- polars, not pandas. Use lazy-safe expressions where the existing code
  already does (`.with_columns`, `.filter`, `pl.col`).
- `gsis_id` is the player join key throughout — not `sleeper_id` (that's
  only used for Sleeper-side roster ownership, not needed tonight).
- `config.POSITIONS = ['QB', 'RB', 'WR', 'TE']` — the only positions this
  pipeline supports. Filter to these anywhere K/DEF might leak in.
- `config.PLAYER_METADATA` now also carries `pfr_id`, `draft_year`,
  `rookie_season` (added for the boom/bust model's snap-share join and
  `yox` feature) — both come through `load_player_metadata`'s existing
  join, no new join logic needed.

## Known gotchas (already hit and fixed once — don't reintroduce)
- polars `.agg()`: use `count()` / `sum()` / `mean()`, never `cum_count()`
  / `cum_sum()` inside an aggregation — the cumulative versions return a
  running value per row, producing a `List` column instead of a scalar,
  which breaks any downstream comparison (`SchemaError` on `>=`).
- `.sort('col')` defaults **ascending**. For "best player at rank N," you
  need `descending=True` — we shipped this bug once already tonight.
- Rank formula is `(league_size * starters_at_pos) + 1` (1-indexed rank),
  but polars list/column indexing is 0-indexed — always index with
  `rank - 1`, not `rank`.
- `.item()` on a multi-row/multi-column DataFrame needs both row and
  column: `df.item(row, 'col')`, not `df.item(row)['col'][0]`.
- `CUTOFF_GAME_AMT` (currently 2) filters players out **before** ranking,
  which can shift who lands at the replacement rank if excluded players
  would otherwise have ranked above it. Tested at both 8 and 2 with no
  meaningful change on this league's real data — accepted as-is, not a
  blocker.

## Phase 1 — required, this is the deliverable
**Task 1: Refactor `eval/replacement_player.py`**
Extract the per-position ranked points table into its own function so
replacement-level and VOR both read from one computation, never two:

```python
def _position_ppgs(position: str, league: League, player_stats: DataFrame) -> DataFrame:
    """Per-player fantasy_ppg + n_games + display_name for one position,
    filtered to n_games >= CUTOFF_GAME_AMT, sorted by fantasy_ppg descending.
    Single source of truth -- _replacement_at_position and get_player_vor
    both call this, neither duplicates the aggregation."""
```
Group by `['gsis_id', 'display_name', 'position']` (currently groups by
`gsis_id` alone, which drops `display_name` — output needs to be
human-readable). `_replacement_at_position` becomes a thin wrapper that
calls this and indexes `replacement_rank - 1`.

**Task 2: Add `get_player_vor(league, previous_season) -> DataFrame`**
in the same module. Combined cross-position table:
`gsis_id, display_name, position, fantasy_ppg, n_games, replacement_ppg, vor`
(`vor = fantasy_ppg - replacement_ppg`), sorted by `vor` descending.

**Task 3: `build_draft_board.py`** at repo root.
- Resolve league via `get_user` → `get_user_leagues` → build `League.from_dict(...)`.
  League to use is a constant at the top of the file (hardcode the
  `league_id` once resolved — don't build a CLI arg parser tonight).
- Call `get_player_vor`.
- Export full result to `outputs/draft_board.csv`.
- Print to console: top 30 overall by `vor`, then top 12 per position.

**Definition of done for Phase 1:** running `python build_draft_board.py`
produces a CSV and console output, ranked by VOR, that a human can
sanity-check a name against (e.g. "is the TE at the top of the board
someone I'd recognize as a real TE1") without errors.

## Phase 2 (revised) — per-position boom/bust classifiers

**Status: data pull, features, and model fitting/evaluation are done
(`model/historical_vor.py`, `model/features.py`, `model/train.py`).
Held-out (2024→2025) Brier/log loss beat a naive constant-rate baseline
for QB/RB/WR; TE is roughly at parity — see README for the numbers.
Remaining: wire `boom_prob`/`bust_prob` into the Phase 1 board (see
Output below) — not done yet.**

### Target
boom = 1 if VOR_season_(N+1) > VOR_season_N + 3 else 0 (k=3, override if
you want a different number). bust_prob = 1 - boom_prob. One binary
classifier per position, not two.

### Features (season-N level, per position)
- previous_season_vor — reuse get_player_vor, parameterized by season
- td_per_game — total TDs (any type) / games played
- yox — season_N - draft_year. Null draft_year (UDFA) -> fall back to
  player's first season with recorded stats, don't drop or error.
- offensive_snap_share — nflreadpy.load_snap_counts(seasons), season-N
  aggregate of offense_pct. Joined on pfr_player_id, NOT gsis_id --
  verify load_ff_playerids() has a PFR id column and add it to
  PLAYER_METADATA if it's not already selected through.
- efficiency — position-specific: yards/target (WR/TE), yards/carry (RB),
  yards/attempt (QB). Four formulas, not one shared column.

### Historical VOR
Apply this league's current scoring settings + roster requirements
retroactively to every season 2019-present. Reuse get_player_vor,
parameterized by season instead of season-agnostic.

### Games-played floor
6+ games required in BOTH season N and season N+1 for a training example
to be included. This is a separate constant from CUTOFF_GAME_AMT (which
stays at 2 for the live board) — don't conflate the two.

### Train/test split
By season boundary, not random. Hold out the most recent transition
(2024→2025) as test. Train on everything earlier.

### Models
Four independent sklearn LogisticRegression, default L2 regularization
left ON. Print per-position training row counts BEFORE fitting anything.

### Evaluation
Brier score + log loss per position on the held-out season. Print them.
No automated accept/reject — user evaluates manually per position.

### Hard requirement: one shared feature function
build_features(stats_df, snap_df, season) must be the exact same function
used for every historical training row AND for scoring the live
2025-season inputs for tonight's draft class. Two slightly-different
versions is silent train/predict skew.

### Output
Same board as Phase 1, boom_prob/bust_prob added per player. Null (not
zero) where a position's model couldn't score a player — don't
silent-fill.