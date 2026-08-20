
import math

import polars as pl
from polars import DataFrame

from config import POSITIONS
from domain.league import League


def find_cliffs(board_df: DataFrame, league: League, top_n: int = 30) -> DataFrame:
    """
    Per position, ranks players by `vor` descending (not ex_value --
    inconsistent across positions right now, would bias TE). Computes
    point-drop to the next-best player at each rank within top_n. Returns
    one row per position: the rank and size of the single largest drop,
    plus that rank converted to an approximate round via
    ceil(rank / league.league_size).

    :param board_df: Anything with display_name, position, vor columns
        (get_player_vor()'s output, or build_draft_board.py's board with
        boom_prob/ex_value already joined on -- those extra columns are
        ignored)
    :param league: League, used for league_size in the round conversion
    :param top_n: How deep into each position's vor ranking to look
    :return: DataFrame with position, cliff_rank, cliff_size, approx_round,
        player_before_cliff, player_after_cliff
    """

    rows = []
    for position in POSITIONS:
        pos_board = board_df.filter(
            pl.col('position') == position
        ).sort('vor', descending=True).head(top_n)

        names = pos_board['display_name'].to_list()
        vors = pos_board['vor'].to_list()

        if len(vors) < 2:
            continue

        drops = [vors[i] - vors[i + 1] for i in range(len(vors) - 1)]
        cliff_idx = max(range(len(drops)), key=lambda i: drops[i])

        cliff_rank = cliff_idx + 1

        rows.append({
            'position': position,
            'cliff_rank': cliff_rank,
            'cliff_size': round(drops[cliff_idx], 4),
            'approx_round': math.ceil(cliff_rank / league.league_size),
            'player_before_cliff': names[cliff_idx],
            'player_after_cliff': names[cliff_idx + 1],
        })

    return pl.DataFrame(rows)
