
import sys

sys.stdout.reconfigure(encoding='utf-8')

from clients.sleeper_client import get_user_leagues, get_user

from domain.league import League

from model.train import train_position_models
from model.predict import score_players


user_id = get_user('jakethompson16')['user_id']
raw_league = get_user_leagues(user_id, '2026')[1]

league = League.from_dict(raw_league)

print("=== Training per-position boom/bust models ===")
models = train_position_models(league)

print("\n=== Inference: 30 random players, 2025-season boom_prob ===")
scored = score_players(league, models, 2025)

print(
    scored
    .select('display_name', 'position', 'previous_season_vor', 'boom_prob')
    .sample(30)
    .sort('boom_prob', descending=True, nulls_last=True)
)
