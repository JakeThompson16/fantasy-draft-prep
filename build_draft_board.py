
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

from clients.sleeper_client import get_user, get_user_leagues
from domain.league import League
from eval.replacement_player import get_player_vor
from eval.scarcity import find_cliffs
from config import POSITIONS
from model.train import train_position_models, ENABLED_POSITIONS
from model.predict import score_players


USERNAME = "jakethompson16"
LEAGUE_ID = "1385766779313201152"  # "Top Tier Fantasy"
PREVIOUS_SEASON = 2025

OUTPUT_PATH = Path("outputs/draft_board.csv")


if __name__ == "__main__":
    user_id = get_user(USERNAME)["user_id"]
    leagues = get_user_leagues(user_id, "2026")
    raw_league = next(l for l in leagues if l["league_id"] == LEAGUE_ID)
    league = League.from_dict(raw_league)

    board = get_player_vor(league, PREVIOUS_SEASON)

    print("=== Positional scarcity cliffs (top 30 by vor per position) ===")
    print(find_cliffs(board, league))

    print("\n=== Training per-position boom/bust models ===")
    models = train_position_models(league)

    scored = score_players(league, models, PREVIOUS_SEASON)
    board = board.join(
        scored.select("gsis_id", "position", "boom_prob", "bust_prob", "ex_value"),
        on=["gsis_id", "position"],
        how="left"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    board.write_csv(OUTPUT_PATH)

    with pl.Config(tbl_rows=30):
        print("\n=== Top 30 overall by VOR ===")
        print(board.head(30))

    with pl.Config(tbl_rows=20):
        print(f"\n=== Top 20 by ex_value ({'/'.join(ENABLED_POSITIONS)}) ===")
        print(
            board
            .filter(pl.col("position").is_in(ENABLED_POSITIONS))
            .sort("ex_value", descending=True, nulls_last=True)
            .head(20)
        )

    for position in POSITIONS:
        with pl.Config(tbl_rows=12):
            print(f"\n=== Top 12 {position} by VOR ===")
            print(
                board.filter(pl.col("position") == position).head(12)
            )
