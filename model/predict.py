
import polars as pl
import nflreadpy as nfl

from clients.nflreadpy.player_data import load_player_stats
from domain.league import League
from eval.replacement_player import get_player_vor
from model.features import build_features
from model.train import FEATURE_COLUMNS


EX_VALUE_K = 0.15
"""ex_value = previous_season_vor + (EX_VALUE_K * boom_prob * replacement_ppg).
Additive, not multiplicative: boom_prob always ADDS to vor, regardless of
its sign -- a multiplicative vor * (1 + k*boom_prob) form would make a
higher boom_prob push a below-replacement player's ex_value further
negative (worse), which inverts what "boom" is supposed to mean. Scaling
the bump by replacement_ppg (rather than a flat point value) keeps it
proportional to the position's own scoring scale. k left at 0.15 --
small on purpose, a deliberate tuning choice, not tuned further here."""


def build_live_features(league: League, season: int) -> pl.DataFrame:
    """Season-`season` feature table for every player, with
    previous_season_vor and replacement_ppg joined in from
    get_player_vor(league, season). Same shape as a training row minus
    the boom label -- this is what gets scored.

    :return: DataFrame with gsis_id, display_name, position,
        FEATURE_COLUMNS, replacement_ppg
    """

    stats_df = load_player_stats(season)
    snap_df = nfl.load_snap_counts([season])

    features = build_features(stats_df, snap_df, season)

    vor_df = get_player_vor(league, season).select(
        'gsis_id', 'position', 'replacement_ppg', pl.col('vor').alias('previous_season_vor')
    )

    return features.join(vor_df, on=['gsis_id', 'position'], how='left')


def score_players(league: League, models: dict, season: int) -> pl.DataFrame:
    """Adds boom_prob, bust_prob, and ex_value to build_live_features()'s
    output. A player at a position with no entry in `models` (e.g. TE --
    see model.train.ENABLED_POSITIONS), or missing ANY required feature,
    gets boom_prob/bust_prob/ex_value = null -- never imputed, never
    dropped from the output.

    :param models: train_position_models(league) output -- only carries
        entries for the positions it was trained on
    :return: DataFrame with gsis_id, display_name, position,
        FEATURE_COLUMNS, boom_prob, bust_prob, ex_value
    """

    live = build_live_features(league, season).with_columns(
        pl.all_horizontal([pl.col(c).is_not_null() for c in FEATURE_COLUMNS])
        .alias('_has_all_features')
    )

    scored_frames = []
    for position in live['position'].unique().to_list():
        pos_live = live.filter(pl.col('position') == position)

        if position not in models:
            scored_frames.append(
                pos_live.with_columns(pl.lit(None, dtype=pl.Float64).alias('boom_prob'))
            )
            continue

        scoreable = pos_live.filter(pl.col('_has_all_features'))
        unscoreable = pos_live.filter(~pl.col('_has_all_features'))

        if scoreable.height > 0:
            scaler = models[position]['scaler']
            model = models[position]['model']

            X_scaled = scaler.transform(scoreable.select(FEATURE_COLUMNS).to_numpy())
            boom_prob = model.predict_proba(X_scaled)[:, 1]

            scoreable = scoreable.with_columns(pl.Series('boom_prob', boom_prob))
        else:
            scoreable = scoreable.with_columns(pl.lit(None, dtype=pl.Float64).alias('boom_prob'))

        unscoreable = unscoreable.with_columns(pl.lit(None, dtype=pl.Float64).alias('boom_prob'))

        scored_frames.append(pl.concat([scoreable, unscoreable]))

    scored = pl.concat(scored_frames).drop('_has_all_features')

    return scored.with_columns(
        (1 - pl.col('boom_prob')).alias('bust_prob'),
        (
            pl.col('previous_season_vor')
            + (EX_VALUE_K * pl.col('boom_prob') * pl.col('replacement_ppg'))
        ).alias('ex_value')
    )
