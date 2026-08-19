
from dataclasses import dataclass, field

from config import POSITIONS
from domain.scoring import ScoringSettings


@dataclass
class League:
    league_id: str
    league_name: str
    season: str
    scoring: ScoringSettings
    league_size: int
    starters: dict[str, int] = field(default_factory=dict)

    def starters_at(self, position: str) -> int:
        """Number of starting roster slots at the given position"""
        return self.starters.get(position, 0)

    @classmethod
    def from_dict(cls, data: dict) -> "League":
        """Builds a League from a raw Sleeper league payload (as returned by
        get_user_leagues()). Roster starter counts are derived by tallying
        data['roster_positions'] for slots matching config.POSITIONS; slots
        like FLEX and BN are not counted toward any single position."""
        starters = {position: 0 for position in POSITIONS}
        for slot in data['roster_positions']:
            if slot in starters:
                starters[slot] += 1

        return cls(
            league_id=data['league_id'],
            league_name=data['name'],
            season=data['season'],
            scoring=ScoringSettings.from_dict(data['scoring_settings']),
            league_size=data['total_rosters'],
            starters=starters
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, League):
            return NotImplemented
        return self.league_id == other.league_id

    def __hash__(self) -> int:
        return hash(self.league_id)
