
from domain.league import League
from eval.replacement_player import get_player_vor

import polars as pl


HISTORICAL_SEASONS = list(range(2019, 2026))
"""2019 through the same previous_season used for the live board (2025)"""

TRAINING_GAMES_FLOOR = 6
"""Games played required in BOTH season N and season N+1 for a transition
to count as a training example. Separate from CUTOFF_GAME_AMT (2, used
for the live board) -- don't conflate the two."""

TEST_TRANSITION = (2024, 2025)
"""Held-out season transition for evaluation. Train on every earlier one."""

BOOM_K = 3
"""boom = 1 if vor_n1 > vor_n + BOOM_K else 0"""


def _season_board(league: League, season: int) -> pl.DataFrame:
    """VOR board for one historical season, this league's current scoring
    and roster requirements applied retroactively (get_player_vor is
    already season-parameterized)."""

    return get_player_vor(league, season).select(
        'gsis_id', 'display_name', 'position', 'vor', 'n_games'
    )


def build_season_transitions(league: League) -> pl.DataFrame:
    """One row per (gsis_id, position) per season transition N -> N+1,
    restricted to players meeting TRAINING_GAMES_FLOOR in both seasons.

    :return: DataFrame with gsis_id, display_name, position, season_n,
        vor_n, n_games_n, vor_n1, n_games_n1, is_test
    """

    frames = []
    for season_n in HISTORICAL_SEASONS[:-1]:
        season_n1 = season_n + 1

        board_n = _season_board(league, season_n)
        board_n1 = _season_board(league, season_n1)

        transition = board_n.join(
            board_n1,
            on=['gsis_id', 'display_name', 'position'],
            how='inner',
            suffix='_n1'
        ).rename(
            {'vor': 'vor_n', 'n_games': 'n_games_n'}
        ).filter(
            (pl.col('n_games_n') >= TRAINING_GAMES_FLOOR) &
            (pl.col('n_games_n1') >= TRAINING_GAMES_FLOOR)
        ).with_columns(
            pl.lit(season_n).alias('season_n'),
            pl.lit((season_n, season_n1) == TEST_TRANSITION).alias('is_test')
        )

        frames.append(transition)

    return pl.concat(frames)


def print_training_row_counts(league: League) -> pl.DataFrame:
    """Per-position train/test row counts after the games floor. Meant to
    be inspected BEFORE any feature engineering or model fitting starts.

    :return: The underlying season-transition DataFrame, for reuse.
    """

    transitions = build_season_transitions(league)

    counts = transitions.group_by(
        ['position', 'is_test']
    ).agg(
        pl.len().alias('rows')
    ).sort(['position', 'is_test'])

    print(counts)

    return transitions


def add_boom_label(transitions: pl.DataFrame) -> pl.DataFrame:
    """Adds the boom target: boom = 1 if vor_n1 > vor_n + BOOM_K else 0."""

    return transitions.with_columns(
        (pl.col('vor_n1') > (pl.col('vor_n') + BOOM_K)).cast(pl.Int8).alias('boom')
    )


def print_class_balance(transitions: pl.DataFrame) -> pl.DataFrame:
    """Per-position boom=1 vs boom=0 counts on the training rows only
    (is_test == False). Meant to be inspected BEFORE build_features() /
    model fitting -- a position with a heavily skewed class balance needs
    that reflected in how its model gets evaluated.

    :param transitions: Output of build_season_transitions(league)
    :return: transitions with the boom label added, for reuse
    """

    labeled = add_boom_label(transitions)

    balance = labeled.filter(
        ~pl.col('is_test')
    ).group_by(
        ['position', 'boom']
    ).agg(
        pl.len().alias('rows')
    ).sort(['position', 'boom'])

    print(balance)

    return labeled
