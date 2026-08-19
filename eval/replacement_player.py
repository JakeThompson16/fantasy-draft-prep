
from clients.nflreadpy.player_data import load_player_stats

from engine.scoring import calculate_points_vectorized

from domain.league import League

import polars as pl
from polars import DataFrame

from config import POSITIONS


CUTOFF_GAME_AMT = 2
"""Must play at least this many games to be eligible"""


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

    position_stats = player_stats.filter(
        pl.col('position') == position
    )

    points_df = calculate_points_vectorized(
        position_stats,
        league.scoring
    )

    points_df = points_df.group_by(
        ['gsis_id']
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

    points_df = points_df.filter(
        pl.col('n_games') >= CUTOFF_GAME_AMT
    ).sort('fantasy_ppg', descending=True)

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