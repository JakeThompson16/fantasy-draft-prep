
from eval.replacement_player import get_replacement_level_ppgs

from clients.sleeper_client import get_user_leagues, get_user

from domain.league import League


user_id = get_user('jakethompson16')['user_id']
raw_league = get_user_leagues(user_id, '2026')[1]

league = League.from_dict(raw_league)

result = get_replacement_level_ppgs(
    league,
    2025
)

print(result)