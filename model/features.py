
import polars as pl


def _verify_pfr_crosswalk(per_player: pl.DataFrame) -> None:
    """Confirms this season's player rows actually carry a usable pfr_id
    before offensive_snap_share relies on it to join snap counts. Raises
    loudly instead of silently producing an all-null feature."""

    non_null = per_player.filter(pl.col('pfr_id').is_not_null()).height
    if non_null == 0:
        raise RuntimeError(
            "No player in this season has a pfr_id -- PFR crosswalk is "
            "missing or broken, offensive_snap_share cannot be joined."
        )


def _snap_share(snap_df: pl.DataFrame, season: int) -> pl.DataFrame:
    """Season-N mean offense_pct per pfr_player_id, regular season only.

    :param snap_df: nfl.load_snap_counts([season]) raw output (may contain
        multiple game_types and, if the caller passed more than one
        season, other seasons too -- both are filtered out here)
    :param season: Season to restrict to
    :return: DataFrame with pfr_player_id, offensive_snap_share
    """

    return snap_df.filter(
        (pl.col('season') == season) & (pl.col('game_type') == 'REG')
    ).group_by('pfr_player_id').agg(
        pl.col('offense_pct').mean().alias('offensive_snap_share')
    )


def _efficiency_expr() -> pl.Expr:
    """Position-specific efficiency: yards/attempt (QB), yards/carry (RB),
    yards/target (WR/TE). Four formulas, not one shared column. Null
    (not 0/inf) when the denominator is 0."""

    return (
        pl.when((pl.col('position') == 'QB') & (pl.col('attempts') > 0))
        .then(pl.col('passing_yards') / pl.col('attempts'))
        .when((pl.col('position') == 'RB') & (pl.col('carries') > 0))
        .then(pl.col('rushing_yards') / pl.col('carries'))
        .when(pl.col('position').is_in(['WR', 'TE']) & (pl.col('targets') > 0))
        .then(pl.col('receiving_yards') / pl.col('targets'))
        .otherwise(None)
        .alias('efficiency')
    )


def build_features(stats_df: pl.DataFrame, snap_df: pl.DataFrame, season: int) -> pl.DataFrame:
    """Season-N, per-player, per-position feature table. This is the exact
    same function used to assemble every historical training row AND to
    score the live season for tonight's draft class -- never fork a
    second version for live scoring, that's train/predict skew.

    previous_season_vor is deliberately NOT produced here: it requires a
    League (scoring settings + roster requirements), and this function
    stays league-agnostic. Callers join get_player_vor(league, season)'s
    `vor` column onto this output separately, keyed on gsis_id + position.

    :param stats_df: load_player_stats(season) output -- one row per
        player per game, regular season only, already carries pfr_id,
        draft_year, rookie_season via PLAYER_METADATA
    :param snap_df: nfl.load_snap_counts([season]) raw output
    :param season: The season these stats/snaps are drawn from (season N,
        whose features predict the boom/bust outcome in season N+1)
    :return: DataFrame with gsis_id, display_name, position, n_games,
        td_per_game, yox, offensive_snap_share, efficiency
    """

    per_player = stats_df.filter(
        pl.col('season') == season
    ).group_by(
        ['gsis_id', 'display_name', 'position', 'pfr_id', 'draft_year', 'rookie_season']
    ).agg(
        pl.len().alias('n_games'),
        (pl.col('passing_tds') + pl.col('rushing_tds') + pl.col('receiving_tds'))
        .sum().alias('total_tds'),
        pl.col('passing_yards').sum().alias('passing_yards'),
        pl.col('attempts').sum().alias('attempts'),
        pl.col('rushing_yards').sum().alias('rushing_yards'),
        pl.col('carries').sum().alias('carries'),
        pl.col('receiving_yards').sum().alias('receiving_yards'),
        pl.col('targets').sum().alias('targets'),
    )

    _verify_pfr_crosswalk(per_player)

    per_player = per_player.with_columns(
        (pl.col('total_tds') / pl.col('n_games')).alias('td_per_game'),
        (season - pl.coalesce(['draft_year', 'rookie_season'])).alias('yox'),
        _efficiency_expr()
    )

    snap_share = _snap_share(snap_df, season)

    per_player = per_player.join(
        snap_share, left_on='pfr_id', right_on='pfr_player_id', how='left'
    )

    if per_player.filter(pl.col('offensive_snap_share').is_not_null()).height == 0:
        print(
            f"WARNING: build_features({season}) matched zero players to "
            f"snap_df on pfr_id -- offensive_snap_share is all-null."
        )

    return per_player.select(
        'gsis_id', 'display_name', 'position', 'n_games', 'td_per_game',
        'yox', 'offensive_snap_share', 'efficiency'
    )
