
import polars as pl
import nflreadpy as nfl

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss

from clients.nflreadpy.player_data import load_player_stats
from domain.league import League
from model.features import build_features
from model.historical_vor import build_season_transitions, add_boom_label


FEATURE_COLUMNS = [
    'previous_season_vor', 'td_per_game', 'yox', 'offensive_snap_share', 'efficiency'
]
"""The 5 features every position's model is fit on, in a fixed order."""

ENABLED_POSITIONS = ['QB', 'RB', 'WR']
"""TE excluded: held-out Brier/log loss (0.0956 / 0.3412) barely beat the
naive constant-rate baseline (0.0961 / 0.3447), too thin a margin to
trust for a live board. TE falls back to plain vor with boom_prob/
bust_prob/ex_value left null -- see README."""


def _season_features(season: int) -> pl.DataFrame:
    """build_features() output for one season, tagged with season_n so it
    can be joined onto build_season_transitions()'s rows."""

    stats_df = load_player_stats(season)
    snap_df = nfl.load_snap_counts([season])

    return build_features(stats_df, snap_df, season).with_columns(
        pl.lit(season).alias('season_n')
    )


def build_training_table(league: League) -> pl.DataFrame:
    """One row per (gsis_id, position, season_n) training example: the 5
    FEATURE_COLUMNS, the boom label, and is_test. Rows with ANY null
    feature are dropped here -- sklearn's LogisticRegression can't take
    nulls, and the spec is drop, not impute.

    :param league: League with scoring settings and roster composition
    :return: DataFrame with gsis_id, display_name, position, season_n,
        FEATURE_COLUMNS, boom, is_test
    """

    transitions = add_boom_label(build_season_transitions(league)).rename(
        {'vor_n': 'previous_season_vor'}
    )

    seasons_needed = transitions['season_n'].unique().to_list()
    features = pl.concat([_season_features(season) for season in seasons_needed])

    table = transitions.join(
        features.select(
            'gsis_id', 'position', 'season_n',
            'td_per_game', 'yox', 'offensive_snap_share', 'efficiency'
        ),
        on=['gsis_id', 'position', 'season_n'],
        how='left'
    )

    before = table.height
    table = table.drop_nulls(subset=FEATURE_COLUMNS)
    dropped = before - table.height
    if dropped:
        print(f"build_training_table: dropped {dropped}/{before} rows with a null feature")

    return table


def train_position_models(league: League) -> dict:
    """Fits one LogisticRegression per position (default L2 regularization),
    on features scaled by a StandardScaler fit on that position's TRAINING
    rows only (post season-boundary split) and applied to both train and
    test. Prints Brier score + log loss per position on the held-out
    (2024->2025) season -- no automated accept/reject, that's a manual call.

    Only ENABLED_POSITIONS get a model -- TE is deliberately excluded (see
    ENABLED_POSITIONS docstring).

    :param league: League with scoring settings and roster composition
    :return: {position: {'scaler', 'model', 'n_train', 'n_test', 'brier',
        'log_loss'}} -- scaler + model are reusable for live scoring.
    """

    table = build_training_table(league)

    results = {}
    metric_rows = []

    for position in ENABLED_POSITIONS:
        pos_table = table.filter(pl.col('position') == position)

        train = pos_table.filter(~pl.col('is_test'))
        test = pos_table.filter(pl.col('is_test'))

        X_train = train.select(FEATURE_COLUMNS).to_numpy()
        y_train = train['boom'].to_numpy()
        X_test = test.select(FEATURE_COLUMNS).to_numpy()
        y_test = test['boom'].to_numpy()

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        boom_prob = model.predict_proba(X_test_scaled)[:, 1]

        brier = brier_score_loss(y_test, boom_prob)
        ll = log_loss(y_test, boom_prob, labels=[0, 1])

        results[position] = {
            'scaler': scaler,
            'model': model,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'brier': brier,
            'log_loss': ll,
        }
        metric_rows.append({
            'position': position,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'brier': round(brier, 4),
            'log_loss': round(ll, 4),
        })

    print(pl.DataFrame(metric_rows))

    return results
