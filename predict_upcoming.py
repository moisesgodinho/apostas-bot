"""Gera palpites futuros +EV usando fixtures e odds atuais."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from clubelo import add_team_strength_features
from config import (
    DEFAULT_LEAGUES,
    DEFAULT_SEASONS,
    NO_VIG_AWAY_COL,
    NO_VIG_DRAW_COL,
    NO_VIG_HOME_COL,
    NO_VIG_OVER_COL,
    NO_VIG_UNDER_COL,
    ODD_AWAY_COL,
    ODD_DRAW_COL,
    ODD_HOME_COL,
    ODD_OVER_COL,
    ODD_UNDER_COL,
    PipelineConfig,
    RESULT_LABELS,
    RESULT_NAME_MAP,
    RESULT_TARGET_COL,
    TARGET_COL,
)
from data_pipeline import (
    add_asian_handicap_features,
    add_elo_features,
    add_elo_features_to_fixtures,
    add_league_context_features,
    add_lineup_features,
    add_match_stats_interaction_features,
    add_match_importance_features,
    add_match_timing_features,
    add_rest_context_features,
    add_rolling_features,
    add_referee_features_to_fixtures,
    add_xg_interaction_features,
    add_target_and_drop_na,
    build_current_elo_ratings,
    build_team_match_table,
    load_football_data,
    prepare_initial_data,
    TEAM_MATCH_STAT_METRICS,
    TEAM_XG_METRICS,
    VENUE_MATCH_STAT_METRICS,
    VENUE_XG_METRICS,
)
from fixtures import (
    bookmaker_is_supported,
    build_bookmaker_odds_long,
    has_over_under_odds,
    has_result_odds,
    implied_probability_from_best_odd,
    load_upcoming_fixtures,
)
from filter_optimizer import load_positive_filter_rules
from modeling import build_time_decay_sample_weights, train_calibrated_xgboost_model
from pipeline import (
    OVER_MARKET_FEATURES,
    RESULT_MARKET_FEATURES,
    prepare_market_dataset,
)
from understat_xg import merge_understat_xg_features


UPCOMING_PREDICTIONS_FILE = "upcoming_predictions.csv"
UPCOMING_ODDS_FILE = "upcoming_odds_by_bookmaker.csv"
UPCOMING_CONTEXT_FILE = "upcoming_context_summary.csv"
FilterRule = dict[str, float | str | None]
EMPTY_UPCOMING_PREDICTION_COLUMNS = [
    "FixtureId",
    "MatchDatetime",
    "MatchDatetimeBR",
    "KickoffTimezone",
    "Liga",
    "HomeTeam",
    "AwayTeam",
    "Market",
    "Selection",
    "ModelProb",
    "ImpliedProb",
    "Edge",
    "BestOdd",
    "BestBookmaker",
    "RequestedBookmaker",
    "PreferredBookmakerAvailable",
    "IsValueBet",
]
EMPTY_UPCOMING_ODDS_COLUMNS = [
    "FixtureId",
    "MatchDatetime",
    "MatchDatetimeBR",
    "KickoffTimezone",
    "Liga",
    "HomeTeam",
    "AwayTeam",
    "Market",
    "Selection",
    "Bookmaker",
    "Odd",
]


def _rolling_mean_before_fixture(
    team_history: pd.DataFrame,
    team: str,
    fixture_datetime: pd.Timestamp,
    metric_col: str,
    window: int,
    side: str | None = None,
) -> float:
    """Calcula a media dos ultimos jogos de um time antes da fixture."""
    mask = (
        team_history["Team"].eq(team)
        & team_history["MatchDatetime"].lt(fixture_datetime)
    )
    if side is not None:
        mask &= team_history["Side"].eq(side)

    values = team_history.loc[mask].sort_values(
        ["MatchDatetime", "MatchId"],
        kind="mergesort",
    )[metric_col].tail(window)
    if len(values) < window:
        return np.nan
    return float(values.mean())


def _days_since_before_fixture(
    team_history: pd.DataFrame,
    team: str,
    fixture_datetime: pd.Timestamp,
    side: str | None = None,
) -> float:
    """Calcula dias desde o ultimo jogo antes da fixture."""
    mask = (
        team_history["Team"].eq(team)
        & team_history["MatchDatetime"].lt(fixture_datetime)
    )
    max_days = 21
    if side is not None:
        mask &= team_history["Side"].eq(side)
        max_days = 35

    previous_matches = team_history.loc[mask].sort_values(
        ["MatchDatetime", "MatchId"],
        kind="mergesort",
    )
    if previous_matches.empty:
        return np.nan

    days = (fixture_datetime - previous_matches.iloc[-1]["MatchDatetime"]).days
    return float(min(max(days, 0), max_days))


def _recent_match_count_before_fixture(
    team_history: pd.DataFrame,
    team: str,
    fixture_datetime: pd.Timestamp,
    days: int,
) -> float:
    """Conta jogos de um time nos ultimos N dias antes da fixture."""
    start_datetime = fixture_datetime - pd.Timedelta(days=days)
    mask = (
        team_history["Team"].eq(team)
        & team_history["MatchDatetime"].lt(fixture_datetime)
        & team_history["MatchDatetime"].ge(start_datetime)
    )
    return float(mask.sum())


def _fixture_form_features(
    team_history: pd.DataFrame,
    fixture: pd.Series,
    window: int,
) -> dict[str, float]:
    """Cria as mesmas features pre-jogo usadas no treino."""
    home_team = fixture["HomeTeam"]
    away_team = fixture["AwayTeam"]
    fixture_datetime = fixture["MatchDatetime"]

    features = {
        "Home_GF_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "GoalsFor",
            window,
        ),
        "Home_GA_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "GoalsAgainst",
            window,
        ),
        "Home_TG_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "TotalGoals",
            window,
        ),
        "Home_Over25_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "IsOver25",
            window,
        ),
        "Home_Points_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "Points",
            window,
        ),
        "Home_GoalDiff_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "GoalDiff",
            window,
        ),
        "Home_WinRate_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "IsWin",
            window,
        ),
        "Home_DrawRate_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "IsDraw",
            window,
        ),
        "Home_LossRate_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "IsLoss",
            window,
        ),
        "Home_CleanSheet_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "CleanSheet",
            window,
        ),
        "Home_FailedToScore_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "FailedToScore",
            window,
        ),
        "Home_Home_GF_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "GoalsFor",
            window,
            side="home",
        ),
        "Home_Home_GA_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "GoalsAgainst",
            window,
            side="home",
        ),
        "Home_Home_TG_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "TotalGoals",
            window,
            side="home",
        ),
        "Home_Home_Over25_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "IsOver25",
            window,
            side="home",
        ),
        "Home_Home_Points_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "Points",
            window,
            side="home",
        ),
        "Home_Home_WinRate_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "IsWin",
            window,
            side="home",
        ),
        "Home_Home_GoalDiff_Roll5": _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            "GoalDiff",
            window,
            side="home",
        ),
        "Home_RestDays": _days_since_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
        ),
        "Home_Home_RestDays": _days_since_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            side="home",
        ),
        "Home_Matches_Last7": _recent_match_count_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            7,
        ),
        "Home_Matches_Last14": _recent_match_count_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            14,
        ),
        "Away_GF_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "GoalsFor",
            window,
        ),
        "Away_GA_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "GoalsAgainst",
            window,
        ),
        "Away_TG_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "TotalGoals",
            window,
        ),
        "Away_Over25_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "IsOver25",
            window,
        ),
        "Away_Points_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "Points",
            window,
        ),
        "Away_GoalDiff_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "GoalDiff",
            window,
        ),
        "Away_WinRate_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "IsWin",
            window,
        ),
        "Away_DrawRate_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "IsDraw",
            window,
        ),
        "Away_LossRate_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "IsLoss",
            window,
        ),
        "Away_CleanSheet_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "CleanSheet",
            window,
        ),
        "Away_FailedToScore_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "FailedToScore",
            window,
        ),
        "Away_Away_GF_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "GoalsFor",
            window,
            side="away",
        ),
        "Away_Away_GA_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "GoalsAgainst",
            window,
            side="away",
        ),
        "Away_Away_TG_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "TotalGoals",
            window,
            side="away",
        ),
        "Away_Away_Over25_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "IsOver25",
            window,
            side="away",
        ),
        "Away_Away_Points_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "Points",
            window,
            side="away",
        ),
        "Away_Away_WinRate_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "IsWin",
            window,
            side="away",
        ),
        "Away_Away_GoalDiff_Roll5": _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            "GoalDiff",
            window,
            side="away",
        ),
        "Away_RestDays": _days_since_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
        ),
        "Away_Away_RestDays": _days_since_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            side="away",
        ),
        "Away_Matches_Last7": _recent_match_count_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            7,
        ),
        "Away_Matches_Last14": _recent_match_count_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            14,
        ),
    }
    for metric in TEAM_MATCH_STAT_METRICS:
        features[f"Home_{metric}_Roll5"] = _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            metric,
            window,
        )
        features[f"Away_{metric}_Roll5"] = _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            metric,
            window,
        )
    for metric in VENUE_MATCH_STAT_METRICS:
        features[f"Home_Home_{metric}_Roll5"] = _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            metric,
            window,
            side="home",
        )
        features[f"Away_Away_{metric}_Roll5"] = _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            metric,
            window,
            side="away",
        )
    for metric in TEAM_XG_METRICS:
        features[f"Home_{metric}_Roll5"] = _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            metric,
            window,
        )
        features[f"Away_{metric}_Roll5"] = _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            metric,
            window,
        )
    for metric in VENUE_XG_METRICS:
        features[f"Home_Home_{metric}_Roll5"] = _rolling_mean_before_fixture(
            team_history,
            home_team,
            fixture_datetime,
            metric,
            window,
            side="home",
        )
        features[f"Away_Away_{metric}_Roll5"] = _rolling_mean_before_fixture(
            team_history,
            away_team,
            fixture_datetime,
            metric,
            window,
            side="away",
        )

    features["Attack_Diff_Roll5"] = (
        features["Home_GF_Roll5"] - features["Away_GF_Roll5"]
    )
    features["Defense_Diff_Roll5"] = (
        features["Home_GA_Roll5"] - features["Away_GA_Roll5"]
    )
    features["Home_Attack_vs_Away_Defense"] = (
        features["Home_GF_Roll5"] - features["Away_GA_Roll5"]
    )
    features["Away_Attack_vs_Home_Defense"] = (
        features["Away_GF_Roll5"] - features["Home_GA_Roll5"]
    )
    features["Total_Attack_Roll5"] = (
        features["Home_GF_Roll5"] + features["Away_GF_Roll5"]
    )
    features["Total_Defense_Roll5"] = (
        features["Home_GA_Roll5"] + features["Away_GA_Roll5"]
    )
    features["Total_Goals_Form_Roll5"] = (
        features["Home_TG_Roll5"] + features["Away_TG_Roll5"]
    )
    features["Over25_Rate_Diff_Roll5"] = (
        features["Home_Over25_Roll5"] - features["Away_Over25_Roll5"]
    )
    features["Venue_Attack_Diff_Roll5"] = (
        features["Home_Home_GF_Roll5"] - features["Away_Away_GF_Roll5"]
    )
    features["Venue_Defense_Diff_Roll5"] = (
        features["Home_Home_GA_Roll5"] - features["Away_Away_GA_Roll5"]
    )
    features["Venue_Total_Goals_Form_Roll5"] = (
        features["Home_Home_TG_Roll5"] + features["Away_Away_TG_Roll5"]
    )
    features["Venue_Over25_Rate_Diff_Roll5"] = (
        features["Home_Home_Over25_Roll5"]
        - features["Away_Away_Over25_Roll5"]
    )
    features["Expected_Total_Goals_Form_Roll5"] = (
        features["Home_GF_Roll5"]
        + features["Away_GA_Roll5"]
        + features["Away_GF_Roll5"]
        + features["Home_GA_Roll5"]
    ) / 2.0
    features["Points_Diff_Roll5"] = (
        features["Home_Points_Roll5"] - features["Away_Points_Roll5"]
    )
    features["GoalDiff_Diff_Roll5"] = (
        features["Home_GoalDiff_Roll5"] - features["Away_GoalDiff_Roll5"]
    )
    features["WinRate_Diff_Roll5"] = (
        features["Home_WinRate_Roll5"] - features["Away_WinRate_Roll5"]
    )
    features["LossRate_Diff_Roll5"] = (
        features["Home_LossRate_Roll5"] - features["Away_LossRate_Roll5"]
    )
    features["CleanSheet_Diff_Roll5"] = (
        features["Home_CleanSheet_Roll5"] - features["Away_CleanSheet_Roll5"]
    )
    features["FailedToScore_Diff_Roll5"] = (
        features["Home_FailedToScore_Roll5"]
        - features["Away_FailedToScore_Roll5"]
    )
    features["Venue_Points_Diff_Roll5"] = (
        features["Home_Home_Points_Roll5"]
        - features["Away_Away_Points_Roll5"]
    )
    features["Venue_WinRate_Diff_Roll5"] = (
        features["Home_Home_WinRate_Roll5"]
        - features["Away_Away_WinRate_Roll5"]
    )
    features["Venue_GoalDiff_Diff_Roll5"] = (
        features["Home_Home_GoalDiff_Roll5"]
        - features["Away_Away_GoalDiff_Roll5"]
    )
    features["RestDays_Diff"] = (
        features["Home_RestDays"] - features["Away_RestDays"]
    )
    features["Venue_RestDays_Diff"] = (
        features["Home_Home_RestDays"] - features["Away_Away_RestDays"]
    )
    return features


def add_upcoming_rolling_features(
    historical_data: pd.DataFrame,
    fixtures: pd.DataFrame,
    feature_cols: Sequence[str],
    window: int,
) -> pd.DataFrame:
    """Adiciona features pre-jogo em fixtures sem usar dados futuros."""
    if fixtures.empty:
        return fixtures.copy()

    history = historical_data.copy().reset_index(drop=True)
    history["MatchId"] = np.arange(len(history))
    team_history = build_team_match_table(history)
    team_history = team_history.dropna(
        subset=["Team", "MatchDatetime", "GoalsFor", "GoalsAgainst"]
    ).copy()

    feature_rows = [
        _fixture_form_features(team_history, fixture, window)
        for _, fixture in fixtures.iterrows()
    ]
    feature_frame = pd.DataFrame(feature_rows, index=fixtures.index)
    featured_fixtures = pd.concat([fixtures.copy(), feature_frame], axis=1)
    featured_fixtures = add_match_timing_features(featured_fixtures)
    featured_fixtures = add_rest_context_features(featured_fixtures)
    featured_fixtures = add_match_stats_interaction_features(featured_fixtures)
    featured_fixtures = add_xg_interaction_features(featured_fixtures)
    featured_fixtures = add_asian_handicap_features(featured_fixtures)

    latest_season_map = (
        historical_data.dropna(subset=["Liga", "Temporada", "MatchDatetime"])
        .sort_values(["Liga", "MatchDatetime"], kind="mergesort")
        .groupby("Liga", sort=False)["Temporada"]
        .last()
        .astype(str)
        .to_dict()
    )
    featured_fixtures = featured_fixtures.copy()
    featured_fixtures["Temporada"] = featured_fixtures["Liga"].map(
        latest_season_map
    ).fillna(featured_fixtures["Temporada"].astype(str))
    featured_fixtures["_IsUpcomingFixture"] = 1.0

    historical_with_flag = historical_data.copy()
    historical_with_flag["_IsUpcomingFixture"] = 0.0
    union_columns = sorted(
        set(historical_with_flag.columns).union(featured_fixtures.columns)
    )
    combined = pd.concat(
        [
            historical_with_flag.reindex(columns=union_columns),
            featured_fixtures.reindex(columns=union_columns),
        ],
        ignore_index=True,
        sort=False,
    )
    combined = add_league_context_features(combined)
    featured_fixtures = combined[
        combined["_IsUpcomingFixture"].eq(1.0)
    ].drop(columns="_IsUpcomingFixture")
    return featured_fixtures.dropna(subset=list(feature_cols)).copy()


def add_upcoming_match_importance_features(
    historical_data: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """Adiciona importancia do jogo futuro com a tabela atual da liga."""
    if fixtures.empty:
        return fixtures.copy()

    feature_frames = []
    required_history_cols = [
        "Liga",
        "Temporada",
        "MatchDatetime",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
    ]
    for league, league_fixtures in fixtures.groupby("Liga", sort=False):
        league_history = historical_data[
            historical_data["Liga"].eq(league)
        ].dropna(subset=["Temporada", "MatchDatetime"])
        if league_history.empty:
            feature_frames.append(league_fixtures)
            continue

        latest_season = (
            league_history.sort_values("MatchDatetime", kind="mergesort")
            .iloc[-1]["Temporada"]
        )
        season_history = league_history[
            league_history["Temporada"].eq(latest_season)
        ][required_history_cols].copy()

        future_rows = league_fixtures.copy()
        future_rows["Temporada"] = latest_season
        future_rows["FTHG"] = np.nan
        future_rows["FTAG"] = np.nan
        future_rows["_FixtureIndex"] = future_rows.index

        combined = pd.concat(
            [
                season_history,
                future_rows[
                    required_history_cols + ["_FixtureIndex"]
                ],
            ],
            ignore_index=True,
            sort=False,
        )
        combined["MatchId"] = np.arange(len(combined))
        scored = add_match_importance_features(combined)
        future_scored = scored[scored["_FixtureIndex"].notna()].copy()
        future_scored["_FixtureIndex"] = future_scored["_FixtureIndex"].astype(int)
        importance_cols = [
            col for col in scored.columns if col.endswith("Importance")
        ] + [
            "Home_PreMatch_Rank",
            "Away_PreMatch_Rank",
            "Rank_Diff",
            "Home_PreMatch_PointsPerGame",
            "Away_PreMatch_PointsPerGame",
            "PointsPerGame_Diff",
            "Home_TitlePressure",
            "Away_TitlePressure",
            "Home_RelegationPressure",
            "Away_RelegationPressure",
            "Importance_Diff",
            "TopClash",
            "SeasonProgress",
        ]
        importance_cols = [
            col for col in dict.fromkeys(importance_cols) if col in scored.columns
        ]
        future_scored = future_scored.set_index("_FixtureIndex")[importance_cols]
        enriched = league_fixtures.join(future_scored, how="left")
        feature_frames.append(enriched)

    return pd.concat(feature_frames).sort_index()


def _positive_class_probability(model, features: pd.DataFrame) -> np.ndarray:
    """Retorna probabilidade da classe positiva em modelos binarios."""
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return probabilities[:, -1]


def _aligned_result_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    """Alinha probabilidades 1X2 na ordem H, D, A."""
    probabilities = model.predict_proba(features)
    aligned = np.zeros((len(features), len(RESULT_LABELS)))

    for source_index, class_value in enumerate(model.classes_):
        aligned[:, int(class_value)] = probabilities[:, source_index]
    return aligned


def _base_prediction_columns(row: pd.Series) -> dict[str, object]:
    """Monta campos comuns das linhas de palpite."""
    return {
        "FixtureId": row["FixtureId"],
        "MatchDatetime": row["MatchDatetime"],
        "MatchDatetimeBR": row.get("MatchDatetimeBR", pd.NaT),
        "KickoffTimezone": row.get("KickoffTimezone", ""),
        "Liga": row["Liga"],
        "HomeTeam": row["HomeTeam"],
        "AwayTeam": row["AwayTeam"],
        "Home_Elo_Pre": row.get("Home_Elo_Pre", np.nan),
        "Away_Elo_Pre": row.get("Away_Elo_Pre", np.nan),
        "Elo_Diff": row.get("Elo_Diff", np.nan),
        "Elo_Home_Adv_Diff": row.get("Elo_Home_Adv_Diff", np.nan),
        "Elo_Expected_Home": row.get("Elo_Expected_Home", np.nan),
        "Home_TeamStrength": row.get("Home_TeamStrength", np.nan),
        "Away_TeamStrength": row.get("Away_TeamStrength", np.nan),
        "TeamStrength_Diff": row.get("TeamStrength_Diff", np.nan),
        "TeamStrength_Expected_Home": row.get(
            "TeamStrength_Expected_Home",
            np.nan,
        ),
        "ClubElo_DataAvailable": row.get("ClubElo_DataAvailable", np.nan),
        "Home_ClubElo_Pre": row.get("Home_ClubElo_Pre", np.nan),
        "Away_ClubElo_Pre": row.get("Away_ClubElo_Pre", np.nan),
        "Home_PreMatch_Rank": row.get("Home_PreMatch_Rank", np.nan),
        "Away_PreMatch_Rank": row.get("Away_PreMatch_Rank", np.nan),
        "MatchImportance": row.get("MatchImportance", np.nan),
        "Home_MatchImportance": row.get("Home_MatchImportance", np.nan),
        "Away_MatchImportance": row.get("Away_MatchImportance", np.nan),
        "SeasonProgress": row.get("SeasonProgress", np.nan),
        "Home_LineupDataAvailable": row.get("Home_LineupDataAvailable", np.nan),
        "Away_LineupDataAvailable": row.get("Away_LineupDataAvailable", np.nan),
        "Home_LineupConfirmed": row.get("Home_LineupConfirmed", np.nan),
        "Away_LineupConfirmed": row.get("Away_LineupConfirmed", np.nan),
        "LineupStrength_Diff": row.get("LineupStrength_Diff", np.nan),
        "MissingStarters_Diff": row.get("MissingStarters_Diff", np.nan),
        "MissingKeyPlayers_Diff": row.get("MissingKeyPlayers_Diff", np.nan),
        "Home_MatchStatsAvailable_Roll5": row.get(
            "Home_MatchStatsAvailable_Roll5",
            np.nan,
        ),
        "Away_MatchStatsAvailable_Roll5": row.get(
            "Away_MatchStatsAvailable_Roll5",
            np.nan,
        ),
        "ShotsOnTarget_Total_Roll5": row.get(
            "ShotsOnTarget_Total_Roll5",
            np.nan,
        ),
        "Corners_Total_Roll5": row.get("Corners_Total_Roll5", np.nan),
        "Over25_ImpliedMove": row.get("Over25_ImpliedMove", np.nan),
        "Result_ClosingAvailable": row.get("Result_ClosingAvailable", np.nan),
        "Odds_Quality_Total": row.get("Odds_Quality_Total", np.nan),
        "Odds_Quality_Result": row.get("Odds_Quality_Result", np.nan),
        "Totals_MaxAvgGap": row.get("Totals_MaxAvgGap", np.nan),
        "Result_MaxAvgGap": row.get("Result_MaxAvgGap", np.nan),
        "Home_Matches_Last7": row.get("Home_Matches_Last7", np.nan),
        "Away_Matches_Last7": row.get("Away_Matches_Last7", np.nan),
        "Home_Matches_Last14": row.get("Home_Matches_Last14", np.nan),
        "Away_Matches_Last14": row.get("Away_Matches_Last14", np.nan),
        "Home_ShortRest": row.get("Home_ShortRest", np.nan),
        "Away_ShortRest": row.get("Away_ShortRest", np.nan),
        "Referee_DataAvailable_Roll20": row.get(
            "Referee_DataAvailable_Roll20",
            np.nan,
        ),
        "Home_xGAvailable_Roll5": row.get("Home_xGAvailable_Roll5", np.nan),
        "Away_xGAvailable_Roll5": row.get("Away_xGAvailable_Roll5", np.nan),
        "Home_xGFor_Roll5": row.get("Home_xGFor_Roll5", np.nan),
        "Away_xGFor_Roll5": row.get("Away_xGFor_Roll5", np.nan),
        "xG_Total_Roll5": row.get("xG_Total_Roll5", np.nan),
        "xG_Expected_Total_Match_Roll5": row.get(
            "xG_Expected_Total_Match_Roll5",
            np.nan,
        ),
        "xG_Diff_Roll5": row.get("xG_Diff_Roll5", np.nan),
    }


def _passes_filters(
    edge: float,
    model_prob: float,
    odd: float,
    min_edge: float,
    min_model_prob: float,
    max_odd: float | None,
) -> bool:
    """Aplica filtros de aposta +EV."""
    if not np.isfinite(edge) or not np.isfinite(model_prob) or not np.isfinite(odd):
        return False
    if edge < min_edge or model_prob < min_model_prob:
        return False
    return max_odd is None or odd <= max_odd


def _default_filter_rule(config: PipelineConfig, market: str) -> FilterRule:
    """Retorna a regra padrao de filtros por mercado."""
    if market == "Over 2.5":
        return {
            "source": "Padrao",
            "edge_threshold": config.edge,
            "min_model_prob": config.min_model_prob,
            "max_odd": config.max_over_odd,
            "eval_roi": np.nan,
            "eval_bets": np.nan,
        }
    if market == "Under 2.5":
        return {
            "source": "Padrao",
            "edge_threshold": config.edge,
            "min_model_prob": config.min_under_prob,
            "max_odd": config.max_under_odd,
            "eval_roi": np.nan,
            "eval_bets": np.nan,
        }
    if market == "Resultado 1X2":
        return {
            "source": "Padrao",
            "edge_threshold": config.edge,
            "min_model_prob": config.min_result_prob,
            "max_odd": config.max_result_odd,
            "eval_roi": np.nan,
            "eval_bets": np.nan,
        }
    return {
        "source": "Padrao",
        "edge_threshold": config.edge,
        "min_model_prob": config.min_win_prob,
        "max_odd": config.max_win_odd,
        "eval_roi": np.nan,
        "eval_bets": np.nan,
    }


def _resolve_filter_rule(
    config: PipelineConfig,
    market: str,
    optimized_rules: dict[str, FilterRule],
) -> FilterRule:
    """Usa regra otimizada se ela passou na avaliacao; senao usa padrao."""
    return optimized_rules.get(market, _default_filter_rule(config, market))


def _training_sample_weights(
    data: pd.DataFrame,
    config: PipelineConfig,
) -> np.ndarray | None:
    """Calcula pesos temporais para modelos finais de palpites futuros."""
    if not config.use_time_decay_weights:
        return None
    return build_time_decay_sample_weights(
        data["MatchDatetime"],
        half_life_days=config.time_decay_half_life_days,
        min_weight=config.min_time_decay_weight,
    )


def _rule_columns(rule: FilterRule) -> dict[str, float | str | None]:
    """Campos de auditoria da regra usada no palpite."""
    return {
        "RuleSource": rule["source"],
        "RuleEdgeThreshold": rule["edge_threshold"],
        "RuleMinModelProb": rule["min_model_prob"],
        "RuleMaxOdd": rule["max_odd"],
        "RuleEvalROI": rule["eval_roi"],
        "RuleEvalBets": rule["eval_bets"],
    }


def _selected_odd_column(selection: str) -> str:
    """Retorna a coluna da odd usada no palpite."""
    mapping = {
        "Over 2.5": "Selected_Odd_Over25",
        "Under 2.5": "Selected_Odd_Under25",
        "Casa": "Selected_Odd_H",
        "Empate": "Selected_Odd_D",
        "Fora": "Selected_Odd_A",
    }
    return mapping[selection]


def _selected_bookmaker_column(selection: str) -> str:
    """Retorna a coluna da casa usada no palpite."""
    mapping = {
        "Over 2.5": "Selected_Bookmaker_Over25",
        "Under 2.5": "Selected_Bookmaker_Under25",
        "Casa": "Selected_Bookmaker_H",
        "Empate": "Selected_Bookmaker_D",
        "Fora": "Selected_Bookmaker_A",
    }
    return mapping[selection]


def build_upcoming_context_summary(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Resume como as odds futuras foram resolvidas."""
    requested = config.preferred_bookmaker or "Melhor disponivel"
    supported = bookmaker_is_supported(config.preferred_bookmaker)

    over_col = _selected_odd_column("Over 2.5")
    under_col = _selected_odd_column("Under 2.5")
    home_col = _selected_odd_column("Casa")
    draw_col = _selected_odd_column("Empate")
    away_col = _selected_odd_column("Fora")

    totals_available = int(
        fixtures[[over_col, under_col]].notna().all(axis=1).sum()
    ) if not fixtures.empty else 0
    result_available = int(
        fixtures[[home_col, draw_col, away_col]].notna().all(axis=1).sum()
    ) if not fixtures.empty else 0

    if fixtures.empty:
        message = "Nenhum jogo futuro foi encontrado para a janela escolhida."
    elif config.preferred_bookmaker and not supported:
        message = (
            "A fonte atual do Football-Data nao traz odds da casa escolhida. "
            "Por isso os palpites ficam vazios usando somente essa casa."
        )
    elif config.preferred_bookmaker and totals_available + result_available == 0:
        message = (
            "A casa escolhida existe no mapeamento, mas nao apareceu nos jogos "
            "carregados para este periodo."
        )
    elif config.preferred_bookmaker:
        message = "Palpites calculados usando apenas a casa escolhida."
    else:
        message = "Palpites calculados usando a melhor odd disponivel."

    return pd.DataFrame(
        [
            {
                "RequestedBookmaker": requested,
                "UsesPreferredBookmaker": bool(config.preferred_bookmaker),
                "RequestedBookmakerSupported": float(supported),
                "FixturesLoaded": int(len(fixtures)),
                "FixturesWithTotalsOdds": totals_available,
                "FixturesWithResultOdds": result_available,
                "PredictionRows": int(len(predictions)),
                "Message": message,
            }
        ]
    )


def score_over25_predictions(
    fixtures: pd.DataFrame,
    model,
    feature_cols: Sequence[str],
    config: PipelineConfig,
    filter_rule: FilterRule,
) -> pd.DataFrame:
    """Gera palpites futuros para Over 2.5."""
    model_cols = list(feature_cols) + OVER_MARKET_FEATURES
    odd_col = _selected_odd_column("Over 2.5")
    bookmaker_col = _selected_bookmaker_column("Over 2.5")
    required_cols = model_cols + [odd_col]
    mask = has_over_under_odds(fixtures) & fixtures[required_cols].notna().all(axis=1)
    scored = fixtures.loc[mask].copy()

    if scored.empty:
        return pd.DataFrame()

    probabilities = _positive_class_probability(model, scored.loc[:, model_cols])
    scored["ModelProb"] = probabilities
    scored["BestOdd"] = scored[odd_col]
    scored["BestBookmaker"] = scored[bookmaker_col]
    scored["ImpliedProb"] = implied_probability_from_best_odd(
        scored,
        odd_col,
    )
    scored["NoVigProb"] = scored[NO_VIG_OVER_COL]
    scored["Edge"] = scored["ModelProb"] - scored["ImpliedProb"]
    scored["IsValueBet"] = [
        _passes_filters(
            edge,
            model_prob,
            odd,
            float(filter_rule["edge_threshold"]),
            float(filter_rule["min_model_prob"]),
            filter_rule["max_odd"],
        )
        for edge, model_prob, odd in zip(
            scored["Edge"],
            scored["ModelProb"],
            scored["BestOdd"],
        )
    ]

    rows = []
    for _, row in scored.iterrows():
        rows.append(
            {
                **_base_prediction_columns(row),
                "Market": "Over 2.5",
                "Selection": "Over 2.5",
                "BestBookmaker": row["BestBookmaker"],
                "BestOdd": row["BestOdd"],
                "ModelProb": row["ModelProb"],
                "ImpliedProb": row["ImpliedProb"],
                "NoVigProb": row["NoVigProb"],
                "Edge": row["Edge"],
                "RequestedBookmaker": row.get(
                    "RequestedBookmaker",
                    "Melhor disponivel",
                ),
                "PreferredBookmakerAvailable": bool(pd.notna(row[odd_col])),
                "IsValueBet": row["IsValueBet"],
                **_rule_columns(filter_rule),
                "Model_Prob_H": np.nan,
                "Model_Prob_D": np.nan,
                "Model_Prob_A": np.nan,
            }
        )

    return pd.DataFrame(rows)


def score_under25_predictions(
    fixtures: pd.DataFrame,
    model,
    feature_cols: Sequence[str],
    config: PipelineConfig,
    filter_rule: FilterRule,
) -> pd.DataFrame:
    """Gera palpites futuros para Under 2.5."""
    model_cols = list(feature_cols) + OVER_MARKET_FEATURES
    odd_col = _selected_odd_column("Under 2.5")
    bookmaker_col = _selected_bookmaker_column("Under 2.5")
    required_cols = model_cols + [odd_col]
    mask = has_over_under_odds(fixtures) & fixtures[required_cols].notna().all(axis=1)
    scored = fixtures.loc[mask].copy()

    if scored.empty:
        return pd.DataFrame()

    over_probabilities = _positive_class_probability(
        model,
        scored.loc[:, model_cols],
    )
    scored["ModelProb"] = 1.0 - over_probabilities
    scored["BestOdd"] = scored[odd_col]
    scored["BestBookmaker"] = scored[bookmaker_col]
    scored["ImpliedProb"] = implied_probability_from_best_odd(
        scored,
        odd_col,
    )
    scored["NoVigProb"] = scored[NO_VIG_UNDER_COL]
    scored["Edge"] = scored["ModelProb"] - scored["ImpliedProb"]
    scored["IsValueBet"] = [
        _passes_filters(
            edge,
            model_prob,
            odd,
            float(filter_rule["edge_threshold"]),
            float(filter_rule["min_model_prob"]),
            filter_rule["max_odd"],
        )
        for edge, model_prob, odd in zip(
            scored["Edge"],
            scored["ModelProb"],
            scored["BestOdd"],
        )
    ]

    rows = []
    for _, row in scored.iterrows():
        rows.append(
            {
                **_base_prediction_columns(row),
                "Market": "Under 2.5",
                "Selection": "Under 2.5",
                "BestBookmaker": row["BestBookmaker"],
                "BestOdd": row["BestOdd"],
                "ModelProb": row["ModelProb"],
                "ImpliedProb": row["ImpliedProb"],
                "NoVigProb": row["NoVigProb"],
                "Edge": row["Edge"],
                "RequestedBookmaker": row.get(
                    "RequestedBookmaker",
                    "Melhor disponivel",
                ),
                "PreferredBookmakerAvailable": bool(pd.notna(row[odd_col])),
                "IsValueBet": row["IsValueBet"],
                **_rule_columns(filter_rule),
                "Model_Prob_H": np.nan,
                "Model_Prob_D": np.nan,
                "Model_Prob_A": np.nan,
            }
        )

    return pd.DataFrame(rows)


def _result_selection_frame(
    scored: pd.DataFrame,
    probabilities: np.ndarray,
    include_draw: bool,
) -> pd.DataFrame:
    """Escolhe a melhor selecao do 1X2 por edge."""
    selections = [
        (0, "Casa", _selected_odd_column("Casa"), _selected_bookmaker_column("Casa")),
        (
            1,
            "Empate",
            _selected_odd_column("Empate"),
            _selected_bookmaker_column("Empate"),
        ),
        (2, "Fora", _selected_odd_column("Fora"), _selected_bookmaker_column("Fora")),
    ]
    if not include_draw:
        selections = [selection for selection in selections if selection[0] != 1]

    candidate_rows: list[dict[str, object]] = []
    for row_index, (_, row) in enumerate(scored.iterrows()):
        best_candidate: dict[str, object] | None = None

        for class_index, selection, odd_col, bookmaker_col in selections:
            odd = row[odd_col]
            implied_prob = 1.0 / odd if np.isfinite(odd) and odd > 1.0 else np.nan
            model_prob = probabilities[row_index, class_index]
            edge = model_prob - implied_prob

            if not np.isfinite(edge):
                continue

            candidate = {
                **_base_prediction_columns(row),
                "Selection": selection,
                "BestBookmaker": row[bookmaker_col],
                "BestOdd": odd,
                "ModelProb": model_prob,
                "ImpliedProb": implied_prob,
                "NoVigProb": row[
                    [NO_VIG_HOME_COL, NO_VIG_DRAW_COL, NO_VIG_AWAY_COL][class_index]
                ],
                "Edge": edge,
                "RequestedBookmaker": row.get(
                    "RequestedBookmaker",
                    "Melhor disponivel",
                ),
                "PreferredBookmakerAvailable": bool(pd.notna(odd)),
                "ResultIndex": class_index,
                "Model_Prob_H": probabilities[row_index, 0],
                "Model_Prob_D": probabilities[row_index, 1],
                "Model_Prob_A": probabilities[row_index, 2],
            }

            if best_candidate is None or edge > float(best_candidate["Edge"]):
                best_candidate = candidate

        if best_candidate is not None:
            candidate_rows.append(best_candidate)

    return pd.DataFrame(candidate_rows)


def score_result_predictions(
    fixtures: pd.DataFrame,
    model,
    feature_cols: Sequence[str],
    config: PipelineConfig,
    filter_rule: FilterRule,
) -> pd.DataFrame:
    """Gera palpites futuros para Resultado Final 1X2."""
    model_cols = list(feature_cols) + RESULT_MARKET_FEATURES
    selected_result_cols = [
        _selected_odd_column("Casa"),
        _selected_odd_column("Empate"),
        _selected_odd_column("Fora"),
    ]
    mask = (
        has_result_odds(fixtures)
        & fixtures[model_cols].notna().all(axis=1)
        & fixtures[selected_result_cols].notna().any(axis=1)
    )
    scored = fixtures.loc[mask].copy()

    if scored.empty:
        return pd.DataFrame()

    probabilities = _aligned_result_probabilities(model, scored.loc[:, model_cols])
    result_frame = _result_selection_frame(
        scored,
        probabilities,
        include_draw=True,
    )
    if result_frame.empty:
        return result_frame

    result_frame["Market"] = "Resultado 1X2"
    result_frame["IsValueBet"] = [
        _passes_filters(
            edge,
            model_prob,
            odd,
            float(filter_rule["edge_threshold"]),
            float(filter_rule["min_model_prob"]),
            filter_rule["max_odd"],
        )
        for edge, model_prob, odd in zip(
            result_frame["Edge"],
            result_frame["ModelProb"],
            result_frame["BestOdd"],
        )
    ]
    for key, value in _rule_columns(filter_rule).items():
        result_frame[key] = value
    return result_frame


def score_win_predictions(
    fixtures: pd.DataFrame,
    model,
    feature_cols: Sequence[str],
    config: PipelineConfig,
    filter_rule: FilterRule,
) -> pd.DataFrame:
    """Gera palpites futuros apenas para vitoria casa/fora."""
    model_cols = list(feature_cols) + RESULT_MARKET_FEATURES
    selected_result_cols = [
        _selected_odd_column("Casa"),
        _selected_odd_column("Fora"),
    ]
    mask = (
        has_result_odds(fixtures)
        & fixtures[model_cols].notna().all(axis=1)
        & fixtures[selected_result_cols].notna().any(axis=1)
    )
    scored = fixtures.loc[mask].copy()

    if scored.empty:
        return pd.DataFrame()

    probabilities = _aligned_result_probabilities(model, scored.loc[:, model_cols])
    win_frame = _result_selection_frame(
        scored,
        probabilities,
        include_draw=False,
    )
    if win_frame.empty:
        return win_frame

    win_frame["Market"] = "Vitoria Casa/Fora"
    win_frame["IsValueBet"] = [
        _passes_filters(
            edge,
            model_prob,
            odd,
            float(filter_rule["edge_threshold"]),
            float(filter_rule["min_model_prob"]),
            filter_rule["max_odd"],
        )
        for edge, model_prob, odd in zip(
            win_frame["Edge"],
            win_frame["ModelProb"],
            win_frame["BestOdd"],
        )
    ]
    for key, value in _rule_columns(filter_rule).items():
        win_frame[key] = value
    return win_frame


def build_historical_model_data(
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, float]]:
    """Carrega historico, cria features e retorna dataset modelavel."""
    raw_data = load_football_data(config.leagues, config.seasons, config.raw_dir)
    prepared_data, *_ = prepare_initial_data(raw_data)
    prepared_data = merge_understat_xg_features(
        prepared_data,
        cache_dir=config.understat_xg_dir,
        enabled=config.use_understat_xg,
        force_refresh=config.force_refresh_understat_xg,
    )
    elo_ratings = build_current_elo_ratings(
        prepared_data,
        initial_rating=config.elo_initial,
        k_factor=config.elo_k_factor,
        home_advantage=config.elo_home_advantage,
    )
    prepared_data = add_elo_features(
        prepared_data,
        initial_rating=config.elo_initial,
        k_factor=config.elo_k_factor,
        home_advantage=config.elo_home_advantage,
    )
    prepared_data = add_team_strength_features(
        prepared_data,
        cache_dir=config.clubelo_cache_dir,
        enabled=config.use_clubelo,
        force_refresh=config.force_refresh_clubelo,
        supported_leagues=config.clubelo_leagues,
        home_advantage=config.elo_home_advantage,
        fallback_rating=config.elo_initial,
    )
    featured_data, feature_cols = add_rolling_features(
        prepared_data,
        window=config.rolling_window,
        feature_profile=config.feature_profile,
        lineup_features_path=config.lineup_features_path,
    )
    model_data = add_target_and_drop_na(featured_data, feature_cols)
    return prepared_data, model_data, feature_cols, elo_ratings


def generate_upcoming_predictions(
    config: PipelineConfig,
    days_ahead: int = 7,
    force_refresh_fixtures: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa o pipeline completo de palpites futuros."""
    fixtures = load_upcoming_fixtures(
        config.raw_dir,
        config.leagues,
        days_ahead=days_ahead,
        force_refresh=force_refresh_fixtures,
        preferred_bookmaker=config.preferred_bookmaker,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = config.output_dir / UPCOMING_PREDICTIONS_FILE
    odds_path = config.output_dir / UPCOMING_ODDS_FILE
    context_path = config.output_dir / UPCOMING_CONTEXT_FILE

    if fixtures.empty:
        pd.DataFrame(columns=EMPTY_UPCOMING_PREDICTION_COLUMNS).to_csv(
            predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(columns=EMPTY_UPCOMING_ODDS_COLUMNS).to_csv(
            odds_path,
            index=False,
            encoding="utf-8-sig",
        )
        build_upcoming_context_summary(
            fixtures,
            pd.DataFrame(columns=EMPTY_UPCOMING_PREDICTION_COLUMNS),
            config,
        ).to_csv(context_path, index=False, encoding="utf-8-sig")
        print("[fixtures] Nenhum jogo futuro encontrado para os filtros atuais.")
        return pd.DataFrame(), pd.DataFrame()

    bookmaker_odds = build_bookmaker_odds_long(fixtures)
    bookmaker_odds.to_csv(odds_path, index=False, encoding="utf-8-sig")

    prepared_data, model_data, feature_cols, elo_ratings = build_historical_model_data(
        config
    )
    fixtures = add_elo_features_to_fixtures(
        fixtures,
        elo_ratings,
        initial_rating=config.elo_initial,
        home_advantage=config.elo_home_advantage,
    )
    fixtures = add_team_strength_features(
        fixtures,
        cache_dir=config.clubelo_cache_dir,
        enabled=config.use_clubelo,
        force_refresh=config.force_refresh_clubelo,
        supported_leagues=config.clubelo_leagues,
        home_advantage=config.elo_home_advantage,
        fallback_rating=config.elo_initial,
    )
    fixtures = add_upcoming_match_importance_features(prepared_data, fixtures)
    fixtures = add_lineup_features(fixtures, config.lineup_features_path)
    fixtures = add_referee_features_to_fixtures(prepared_data, fixtures)
    featured_fixtures = add_upcoming_rolling_features(
        prepared_data,
        fixtures,
        feature_cols,
        config.rolling_window,
    )

    if featured_fixtures.empty:
        pd.DataFrame(columns=EMPTY_UPCOMING_PREDICTION_COLUMNS).to_csv(
            predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
        build_upcoming_context_summary(
            fixtures,
            pd.DataFrame(columns=EMPTY_UPCOMING_PREDICTION_COLUMNS),
            config,
        ).to_csv(context_path, index=False, encoding="utf-8-sig")
        print("[features] Nenhuma fixture com historico suficiente para prever.")
        return pd.DataFrame(), bookmaker_odds

    optimized_rules = (
        load_positive_filter_rules(
            config.output_dir,
            min_eval_roi=config.min_optimized_eval_roi,
            min_eval_bets=config.min_optimized_eval_bets,
        )
        if config.use_optimized_filters_for_upcoming
        else {}
    )
    if optimized_rules:
        markets_text = ", ".join(sorted(optimized_rules))
        print(f"[filtros] Usando regras otimizadas para: {markets_text}")
    else:
        print("[filtros] Usando filtros padrao para todos os mercados.")

    prediction_frames: list[pd.DataFrame] = []

    if "over25" in config.markets or "under25" in config.markets:
        over_data, over_feature_cols = prepare_market_dataset(
            model_data,
            feature_cols,
            TARGET_COL,
            OVER_MARKET_FEATURES,
        )
        if not over_data.empty:
            print("[modelo] Treinando modelo final Over/Under 2.5...")
            over_model = train_calibrated_xgboost_model(
                over_data.loc[:, over_feature_cols],
                over_data[TARGET_COL],
                config.calibration_size,
                config.calibration_method,
                config.xgb_tuning_trials,
                config.xgb_tuning_validation_size,
                sample_weight=_training_sample_weights(over_data, config),
            )
            if "over25" in config.markets:
                prediction_frames.append(
                    score_over25_predictions(
                        featured_fixtures,
                        over_model,
                        feature_cols,
                        config,
                        _resolve_filter_rule(config, "Over 2.5", optimized_rules),
                    )
                )
            if "under25" in config.markets:
                prediction_frames.append(
                    score_under25_predictions(
                        featured_fixtures,
                        over_model,
                        feature_cols,
                        config,
                        _resolve_filter_rule(config, "Under 2.5", optimized_rules),
                    )
                )
        else:
            print("[aviso] Over/Under 2.5 sem dados/odds suficientes para prever.")

    if "result" in config.markets or "win" in config.markets:
        result_data, result_feature_cols = prepare_market_dataset(
            model_data,
            feature_cols,
            RESULT_TARGET_COL,
            RESULT_MARKET_FEATURES,
        )
        if not result_data.empty:
            print("[modelo] Treinando modelo final 1X2/Vitoria...")
            result_model = train_calibrated_xgboost_model(
                result_data.loc[:, result_feature_cols],
                result_data[RESULT_TARGET_COL],
                config.calibration_size,
                config.calibration_method,
                config.xgb_tuning_trials,
                config.xgb_tuning_validation_size,
                sample_weight=_training_sample_weights(result_data, config),
            )
            if "result" in config.markets:
                prediction_frames.append(
                    score_result_predictions(
                        featured_fixtures,
                        result_model,
                        feature_cols,
                        config,
                        _resolve_filter_rule(
                            config,
                            "Resultado 1X2",
                            optimized_rules,
                        ),
                    )
                )
            if "win" in config.markets:
                prediction_frames.append(
                    score_win_predictions(
                        featured_fixtures,
                        result_model,
                        feature_cols,
                        config,
                        _resolve_filter_rule(
                            config,
                            "Vitoria Casa/Fora",
                            optimized_rules,
                        ),
                    )
                )
        else:
            print("[aviso] 1X2/Vitoria sem dados/odds suficientes para prever.")

    prediction_frames = [frame for frame in prediction_frames if not frame.empty]
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions = predictions.sort_values(
            ["IsValueBet", "MatchDatetime", "Edge"],
            ascending=[False, True, False],
            kind="mergesort",
        )
    else:
        predictions = pd.DataFrame()

    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    build_upcoming_context_summary(
        fixtures,
        predictions,
        config,
    ).to_csv(context_path, index=False, encoding="utf-8-sig")
    print(f"[saida] Palpites futuros salvos em: {predictions_path}")
    print(f"[saida] Odds por casa salvas em: {odds_path}")
    print(f"[saida] Resumo da casa usada salvo em: {context_path}")

    if not predictions.empty:
        value_bets = int(predictions["IsValueBet"].sum())
        print(f"[palpites] Linhas geradas: {len(predictions):,}")
        print(f"[palpites] Apostas +EV encontradas: {value_bets:,}")

    return predictions, bookmaker_odds


def parse_args() -> tuple[PipelineConfig, int, bool]:
    """Le argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Gera palpites futuros usando fixtures do Football-Data."
    )
    parser.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES)
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=["over25", "under25", "result", "win", "all"],
        default=["all"],
    )
    parser.add_argument("--days-ahead", type=int, default=7)
    parser.add_argument("--raw-dir", default="raw_data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--lineup-features-path",
        default="raw_data/lineup_features.csv",
        help=(
            "CSV opcional com escalações/desfalques por time. Use vazio "
            "para desativar."
        ),
    )
    parser.add_argument(
        "--understat-xg-dir",
        default="raw_data/understat_xg",
        help="Pasta de cache para xG historico do Understat.",
    )
    parser.add_argument(
        "--no-understat-xg",
        action="store_true",
        help="Desativa a coleta/uso de xG do Understat.",
    )
    parser.add_argument(
        "--force-refresh-understat-xg",
        action="store_true",
        help="Forca novo download do cache de xG do Understat.",
    )
    parser.add_argument(
        "--use-clubelo",
        action="store_true",
        help=(
            "Usa ClubElo como fonte externa de forca dos times. Quando "
            "indisponivel, o Elo interno continua sendo usado."
        ),
    )
    parser.add_argument(
        "--clubelo-cache-dir",
        default="raw_data/clubelo",
        help="Pasta de cache para historicos ClubElo por clube.",
    )
    parser.add_argument(
        "--force-refresh-clubelo",
        action="store_true",
        help="Forca novo download do cache ClubElo.",
    )
    parser.add_argument("--rolling-window", type=int, default=5)
    parser.add_argument("--calibration-size", type=float, default=0.20)
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
    )
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--edge", type=float, default=0.05)
    parser.add_argument("--min-model-prob", type=float, default=0.55)
    parser.add_argument("--max-over-odd", type=float, default=1.80)
    parser.add_argument("--min-under-prob", type=float, default=0.55)
    parser.add_argument("--max-under-odd", type=float, default=1.80)
    parser.add_argument("--min-result-prob", type=float, default=0.48)
    parser.add_argument("--max-result-odd", type=float, default=2.50)
    parser.add_argument("--min-win-prob", type=float, default=0.50)
    parser.add_argument("--max-win-odd", type=float, default=2.50)
    parser.add_argument("--elo-initial", type=float, default=1500.0)
    parser.add_argument("--elo-k-factor", type=float, default=20.0)
    parser.add_argument("--elo-home-advantage", type=float, default=65.0)
    parser.add_argument(
        "--feature-profile",
        choices=["base", "extended"],
        default="extended",
        help=(
            "Perfil de features do modelo. 'base' usa sinais estaveis; "
            "'extended' inclui estatisticas de jogo, forma, descanso, "
            "importancia e movimento de odds."
        ),
    )
    parser.add_argument(
        "--xgb-tuning-trials",
        type=int,
        default=0,
        help=(
            "Numero de perfis XGBoost testados em validacao temporal. "
            "Use 0 para desativar."
        ),
    )
    parser.add_argument(
        "--xgb-tuning-validation-size",
        type=float,
        default=0.20,
        help="Percentual do treino base usado para escolher hiperparametros.",
    )
    parser.add_argument(
        "--no-time-decay-weights",
        action="store_true",
        help="Desativa pesos temporais no treino dos modelos finais.",
    )
    parser.add_argument(
        "--time-decay-half-life-days",
        type=float,
        default=540.0,
        help="Meia-vida dos pesos temporais em dias.",
    )
    parser.add_argument(
        "--min-time-decay-weight",
        type=float,
        default=0.20,
        help="Peso minimo bruto antes da normalizacao temporal.",
    )
    parser.add_argument(
        "--ignore-optimized-filters",
        action="store_true",
        help="Usa filtros padrao mesmo se houver regras otimizadas positivas.",
    )
    parser.add_argument(
        "--preferred-bookmaker",
        default=None,
        help=(
            "Casa de aposta que deve ser usada nos palpites futuros. "
            "Ex.: Betano, Bet365, Pinnacle."
        ),
    )
    parser.add_argument(
        "--min-optimized-eval-roi",
        type=float,
        default=0.0,
        help="ROI minimo na avaliacao para liberar filtro otimizado futuro.",
    )
    parser.add_argument(
        "--min-optimized-eval-bets",
        type=int,
        default=10,
        help="Volume minimo de apostas na avaliacao para liberar filtro otimizado.",
    )
    parser.add_argument(
        "--force-refresh-fixtures",
        action="store_true",
        help="Forca novo download do CSV de fixtures.",
    )

    args = parser.parse_args()
    markets = (
        ["over25", "under25", "result", "win"]
        if "all" in args.markets
        else args.markets
    )
    max_over_odd = args.max_over_odd if args.max_over_odd > 0 else None
    max_under_odd = args.max_under_odd if args.max_under_odd > 0 else None
    max_result_odd = args.max_result_odd if args.max_result_odd > 0 else None
    max_win_odd = args.max_win_odd if args.max_win_odd > 0 else None

    config = PipelineConfig(
        leagues=args.leagues,
        seasons=args.seasons,
        markets=markets,
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        lineup_features_path=(
            Path(args.lineup_features_path)
            if args.lineup_features_path
            else None
        ),
        understat_xg_dir=Path(args.understat_xg_dir),
        use_understat_xg=not args.no_understat_xg,
        force_refresh_understat_xg=args.force_refresh_understat_xg,
        clubelo_cache_dir=Path(args.clubelo_cache_dir),
        use_clubelo=args.use_clubelo,
        force_refresh_clubelo=args.force_refresh_clubelo,
        rolling_window=args.rolling_window,
        calibration_size=args.calibration_size,
        calibration_method=args.calibration_method,
        walk_forward_splits=0,
        stake=args.stake,
        edge=args.edge,
        min_model_prob=args.min_model_prob,
        max_over_odd=max_over_odd,
        min_under_prob=args.min_under_prob,
        max_under_odd=max_under_odd,
        min_result_prob=args.min_result_prob,
        max_result_odd=max_result_odd,
        min_win_prob=args.min_win_prob,
        max_win_odd=max_win_odd,
        elo_initial=args.elo_initial,
        elo_k_factor=args.elo_k_factor,
        elo_home_advantage=args.elo_home_advantage,
        feature_profile=args.feature_profile,
        xgb_tuning_trials=args.xgb_tuning_trials,
        xgb_tuning_validation_size=args.xgb_tuning_validation_size,
        use_time_decay_weights=not args.no_time_decay_weights,
        time_decay_half_life_days=args.time_decay_half_life_days,
        min_time_decay_weight=args.min_time_decay_weight,
        use_optimized_filters_for_upcoming=not args.ignore_optimized_filters,
        min_optimized_eval_roi=args.min_optimized_eval_roi,
        min_optimized_eval_bets=args.min_optimized_eval_bets,
        preferred_bookmaker=args.preferred_bookmaker,
    )
    return config, args.days_ahead, args.force_refresh_fixtures


def main() -> None:
    """Ponto de entrada do script."""
    config, days_ahead, force_refresh_fixtures = parse_args()
    generate_upcoming_predictions(
        config,
        days_ahead=days_ahead,
        force_refresh_fixtures=force_refresh_fixtures,
    )


if __name__ == "__main__":
    main()
