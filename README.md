# Fantasy Football Draft Prep
*Finding target players and exploiting positional value*

Fantasy football drafts reward players who are worth more than whatever's
available on waivers after the draft — "replacement level." A league that
starts two RBs and three WRs values those positions differently than one
that starts three RBs and one WR, and a league that awards a point per
reception values pass-catching backs differently than one that doesn't.
Generic rankings ignore all of this. This project computes replacement-level
value *for a specific league's actual settings*, pulled live from that
league, so the numbers reflect the roster and scoring you're actually
drafting into.

Given a Sleeper league, the tool:

1. Reads that league's scoring rules and starting roster requirements
   directly from the Sleeper API.
2. Pulls the previous season's play-by-play-derived stats for every
   rostered player (via `nflreadpy`).
3. Converts raw stats into fantasy points using the league's own scoring
   weights.
4. Finds the "replacement level" player at each position — the best player
   who would *not* be a starter in a league of that size — and reports
   their points-per-game.

That replacement-level PPG is the baseline for evaluating draft picks: a
player's real value is how far above replacement level they project, not
their raw point total.

## Boom/bust signal

Beyond VOR, four independent per-position logistic regressions (QB/RB/WR/TE)
flag players likely to beat their own replacement-level trajectory the
following season. They're trained by retroactively applying this league's
current scoring and roster rules to every season since 2019 (holding out
the 2024→2025 transition for evaluation) on five features per player-season:
prior-season VOR, TDs per game, years of experience, offensive snap share,
and position-specific yards efficiency (yards/attempt, /carry, or /target).

Held-out performance (Brier score / log loss, lower is better, vs. a naive
constant-rate baseline):

| Position | Brier (model / naive) | Log loss (model / naive) |
|---|---|---|
| QB | 0.170 / 0.209 | 0.500 / 0.610 |
| RB | 0.110 / 0.119 | 0.376 / 0.407 |
| WR | 0.160 / 0.179 | 0.483 / 0.546 |
| TE | 0.096 / 0.096 | 0.341 / 0.345 |

QB/RB/WR beat the baseline by a real margin; TE is roughly at parity, which
tracks with TE having the thinnest positive-class sample of the four.
