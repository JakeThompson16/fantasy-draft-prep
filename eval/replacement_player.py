
from clients.nflreadpy.player_data import load_player_stats

from engine.scoring import calculate_points_vectorized

from domain.league import League

import polars as pl
from polars import DataFrame

from config import POSITIONS


CUTOFF_GAME_AMT = 2
"""Must play at least this many games to be eligible"""


def _position_ppgs(
        position: str,
        league: League,
        player_stats: DataFrame) -> DataFrame:
    """Per-player fantasy_ppg + n_games + display_name for one position,
    filtered to n_games >= CUTOFF_GAME_AMT, sorted by fantasy_ppg descending.
    Single source of truth -- _replacement_at_position and get_player_vor
    both call this, neither duplicates the aggregation.

    :param position: Position to compute per-player ppgs for
    :param league: League with scoring settings and roster composition
    :param player_stats: nflreadpy load_player_stats() DataFrame
    :return: DataFrame with gsis_id, display_name, position, fantasy_ppg, n_games
    """

    position_stats = player_stats.filter(
        pl.col('position') == position
    )

    points_df = calculate_points_vectorized(
        position_stats,
        league.scoring
    )

    points_df = points_df.group_by(
        ['gsis_id', 'display_name', 'position']
    ).agg(

        # Points per game
        pl.col('fantasy_points')
        .mean()
        .alias('fantasy_ppg'),

        # Number of games
        pl.col('fantasy_points')
        .count()
        .alias('n_games')
    )

    return points_df.filter(
        pl.col('n_games') >= CUTOFF_GAME_AMT
    ).sort('fantasy_ppg', descending=True)


def _replacement_at_position(
        position: str,
        league: League,
        player_stats: DataFrame) -> float:
    """
    :param position: Position to find replacement level player for
    :param league: League with scoring settings and roster composition
    :param player_stats: nflreadpy load_player_stats() DataFrame
    :return: Replacement player ppg @ position
    """

    replacement_rank = (league.league_size * league.starters_at(position)) + 1

    points_df = _position_ppgs(position, league, player_stats)

    return points_df['fantasy_ppg'][replacement_rank - 1]


def get_replacement_level_ppgs(
        league: League,
        previous_season: int) -> dict:
    """
    :param league: League with scoring settings and roster composition
    :param previous_season: Previous season to construct draft data from
    :return: dict with ppg of replacement level player given leagues scoring settings
    """

    result: dict = {}

    stats_df = load_player_stats(previous_season)

    for position in POSITIONS:
        result[position] = _replacement_at_position(
            position=position,
            league=league,
            player_stats=stats_df
        )

    return result


def get_player_vor(
        league: League,
        previous_season: int) -> DataFrame:
    """
    :param league: League with scoring settings and roster composition
    :param previous_season: Previous season to construct draft data from
    :return: DataFrame with gsis_id, display_name, position, fantasy_ppg, n_games,
        replacement_ppg, vor -- combined across positions, sorted by vor descending
    """

    stats_df = load_player_stats(previous_season)

    position_frames = []
    for position in POSITIONS:
        position_df = _position_ppgs(position, league, stats_df)

        replacement_rank = (league.league_size * league.starters_at(position)) + 1
        replacement_ppg = position_df['fantasy_ppg'][replacement_rank - 1]

        position_frames.append(
            position_df.with_columns(
                pl.lit(replacement_ppg).alias('replacement_ppg')
            )
        )

    vor_df = pl.concat(position_frames).with_columns(
        (pl.col('fantasy_ppg') - pl.col('replacement_ppg')).alias('vor')
    )

    return vor_df.sort('vor', descending=True)
