
import sys
from pathlib import Path

import polars as pl

from config import POSITIONS
from domain.league import League
from domain.scoring import ScoringSettings
from eval.scarcity import find_cliffs

sys.stdout.reconfigure(encoding="utf-8")


BOARD_PATH = Path("outputs/draft_board.csv")
CHEAT_SHEET_PATH = Path("shell_cheat_sheet.txt")

LEAGUE_SIZE = 12
"""Hardcoded rather than re-resolved via Sleeper -- this shell only reads
the already-validated outputs/draft_board.csv, it doesn't touch the API."""

DEFAULT_BOARD_N = 15
SORT_COLUMNS = {"vor", "ex_value"}
DISPLAY_COLUMNS = [
    "position_rank", "display_name", "position", "vor", "ex_value", "boom_prob", "n_games"
]


def _load_board() -> pl.DataFrame:
    """Loads outputs/draft_board.csv once and adds a static position_rank
    (1 = best vor at that position). Never recomputed after this -- the
    shell is a read-only viewer over tonight's validated board."""

    if not BOARD_PATH.exists():
        raise FileNotFoundError(f"{BOARD_PATH} not found -- run build_draft_board.py first.")

    board = pl.read_csv(BOARD_PATH)

    return board.with_columns(
        pl.col("vor").rank(method="ordinal", descending=True).over("position").alias("position_rank")
    )


def _stub_league(league_size: int) -> League:
    """Only league_size matters to find_cliffs()'s approx_round calc --
    everything else on League is irrelevant here, so it's left blank
    rather than re-fetched from Sleeper."""

    return League(
        league_id="", league_name="", season="",
        scoring=ScoringSettings(), league_size=league_size, starters={}
    )


def _match_players(board: pl.DataFrame, query: str) -> pl.DataFrame:
    return board.filter(
        pl.col("display_name").str.to_lowercase().str.contains(query.lower(), literal=True)
    )


def _resolve_one_player(board: pl.DataFrame, query: str) -> dict | None:
    """Case-insensitive substring match. Prints matches and returns None
    if there isn't exactly one -- callers should bail out on None."""

    matches = _match_players(board, query)

    if matches.height == 0:
        print(f"no player matching '{query}'")
        return None

    if matches.height > 1:
        print(f"{matches.height} matches for '{query}', be more specific:")
        for name in matches["display_name"].to_list():
            print(f"  - {name}")
        return None

    return matches.row(0, named=True)


def cmd_board(state: dict, args: list) -> None:
    if not args:
        print("usage: board <position|all> [n] [--by vor|ex_value] [--all]")
        return

    position = args[0].upper()
    rest = args[1:]

    ignore_drafted = "--all" in rest
    rest = [a for a in rest if a != "--all"]

    sort_col = "vor"
    if "--by" in rest:
        idx = rest.index("--by")
        if idx + 1 >= len(rest):
            print("usage: board <position|all> [n] [--by vor|ex_value] [--all]")
            return
        sort_col = rest[idx + 1]
        if sort_col not in SORT_COLUMNS:
            print(f"--by must be one of {sorted(SORT_COLUMNS)}")
            return
        rest = rest[:idx] + rest[idx + 2:]

    n = DEFAULT_BOARD_N
    if rest:
        if not rest[0].isdigit():
            print(f"n must be a number, got '{rest[0]}'")
            return
        n = int(rest[0])

    board = state["board"]
    if not ignore_drafted:
        board = board.filter(~pl.col("gsis_id").is_in(state["drafted"]))

    if position != "ALL":
        if position not in POSITIONS:
            print(f"unknown position '{position}' -- expected one of {POSITIONS} or all")
            return
        board = board.filter(pl.col("position") == position)

    board = board.sort(sort_col, descending=True, nulls_last=True).head(n)

    with pl.Config(tbl_rows=n):
        print(board.select(DISPLAY_COLUMNS))


def cmd_cliffs(state: dict, args: list) -> None:
    board = state["board"].filter(~pl.col("gsis_id").is_in(state["drafted"]))
    print(find_cliffs(board, state["league"]))


def cmd_player(state: dict, args: list) -> None:
    if not args:
        print("usage: player <name>")
        return

    row = _resolve_one_player(state["board"], " ".join(args))
    if row is None:
        return

    for col in DISPLAY_COLUMNS:
        print(f"{col}: {row[col]}")
    print(f"bust_prob: {row['bust_prob']}")
    print(f"drafted: {row['gsis_id'] in state['drafted']}")


def cmd_drafted(state: dict, args: list) -> None:
    if not args:
        print("usage: drafted <name>")
        return

    row = _resolve_one_player(state["board"], " ".join(args))
    if row is None:
        return

    state["drafted"].add(row["gsis_id"])
    print(f"marked drafted: {row['display_name']} ({row['position']})")


def cmd_undraft(state: dict, args: list) -> None:
    if not args:
        print("usage: undraft <name>")
        return

    row = _resolve_one_player(state["board"], " ".join(args))
    if row is None:
        return

    state["drafted"].discard(row["gsis_id"])
    print(f"undrafted: {row['display_name']} ({row['position']})")


def cmd_help(state: dict, args: list) -> None:
    print(CHEAT_SHEET_PATH.read_text(encoding="utf-8"))


def cmd_exit(state: dict, args: list) -> None:
    raise SystemExit


COMMANDS = {
    "board": cmd_board,
    "cliffs": cmd_cliffs,
    "player": cmd_player,
    "drafted": cmd_drafted,
    "undraft": cmd_undraft,
    "help": cmd_help,
    "exit": cmd_exit,
    "quit": cmd_exit,
}


def main() -> None:
    board = _load_board()
    state = {
        "board": board,
        "league": _stub_league(LEAGUE_SIZE),
        "drafted": set(),
    }

    print(f"Loaded {board.height} players from {BOARD_PATH}. Type 'help' for commands.")

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break

        if not line:
            continue

        parts = line.split()
        cmd, args = parts[0].lower(), parts[1:]

        handler = COMMANDS.get(cmd)
        if handler is None:
            print(f"unknown command '{cmd}' -- type 'help' for commands")
            continue

        try:
            handler(state, args)
        except SystemExit:
            break
        except Exception as e:
            print(f"error: {e}")


if __name__ == "__main__":
    main()
