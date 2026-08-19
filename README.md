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
