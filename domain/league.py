
from dataclasses import dataclass

from domain.scoring import ScoringSettings


@dataclass
class League:
    league_id: str
    league_name: str
    season: str
    scoring: ScoringSettings

    def __eq__(self, other) -> bool:
        if not isinstance(other, League):
            return NotImplemented
        return self.league_id == other.league_id

    def __hash__(self) -> int:
        return hash(self.league_id)
