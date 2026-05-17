"""Gera palpites futuros +EV usando fixtures e odds atuais."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from config import (
    DEFAULT_LEAGUES,
    DEFAULT_SEASONS,
    NO_VIG_AWAY_COL,
    NO_VIG_DRAW_COL,
    NO_VIG_HOME_COL,
    NO_VIG_OVER_COL,
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
    add_rolling_features,
    add_target_and_drop_na,
    build_team_match_table,
    load_football_data,
    prepare_initial_data,
)
from fixtures import (
    build_bookmaker_odds_long,
    has_over_under_odds,
    has_result_odds,
    implied_probability_from_best_odd,
    load_upcoming_fixtures,
)
from modeling import train_calibrated_xgboost_model
from pipeline import (
    OVER_MARKET_FEATURES,
    RESULT_MARKET_FEATURES,
    prepare_market_dataset,
)


UPCOMING_PREDICTIONS_FILE = "upcoming_predictions.csv"
UPCOMING_ODDS_FILE = "upcoming_odds_by_bookmaker.csv"


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
    }

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
    return featured_fixtures.dropna(subset=list(feature_cols)).copy()


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


def score_over25_predictions(
    fixtures: pd.DataFrame,
    model,
    feature_cols: Sequence[str],
    config: PipelineConfig,
) -> pd.DataFrame:
    """Gera palpites futuros para Over 2.5."""
    model_cols = list(feature_cols) + OVER_MARKET_FEATURES
    required_cols = model_cols + ["Best_Odd_Over25"]
    mask = has_over_under_odds(fixtures) & fixtures[required_cols].notna().all(axis=1)
    scored = fixtures.loc[mask].copy()

    if scored.empty:
        return pd.DataFrame()

    probabilities = _positive_class_probability(model, scored.loc[:, model_cols])
    scored["ModelProb"] = probabilities
    scored["BestOdd"] = scored["Best_Odd_Over25"]
    scored["BestBookmaker"] = scored["Best_Bookmaker_Over25"]
    scored["ImpliedProb"] = implied_probability_from_best_odd(
        scored,
        "Best_Odd_Over25",
    )
    scored["NoVigProb"] = scored[NO_VIG_OVER_COL]
    scored["Edge"] = scored["ModelProb"] - scored["ImpliedProb"]
    scored["IsValueBet"] = [
        _passes_filters(
            edge,
            model_prob,
            odd,
            config.edge,
            config.min_model_prob,
            config.max_over_odd,
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
                "IsValueBet": row["IsValueBet"],
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
        (0, "Casa", "Best_Odd_H", "Best_Bookmaker_H"),
        (1, "Empate", "Best_Odd_D", "Best_Bookmaker_D"),
        (2, "Fora", "Best_Odd_A", "Best_Bookmaker_A"),
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
) -> pd.DataFrame:
    """Gera palpites futuros para Resultado Final 1X2."""
    model_cols = list(feature_cols) + RESULT_MARKET_FEATURES
    mask = has_result_odds(fixtures) & fixtures[model_cols].notna().all(axis=1)
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
            config.edge,
            config.min_result_prob,
            config.max_result_odd,
        )
        for edge, model_prob, odd in zip(
            result_frame["Edge"],
            result_frame["ModelProb"],
            result_frame["BestOdd"],
        )
    ]
    return result_frame


def score_win_predictions(
    fixtures: pd.DataFrame,
    model,
    feature_cols: Sequence[str],
    config: PipelineConfig,
) -> pd.DataFrame:
    """Gera palpites futuros apenas para vitoria casa/fora."""
    model_cols = list(feature_cols) + RESULT_MARKET_FEATURES
    mask = has_result_odds(fixtures) & fixtures[model_cols].notna().all(axis=1)
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
            config.edge,
            config.min_win_prob,
            config.max_win_odd,
        )
        for edge, model_prob, odd in zip(
            win_frame["Edge"],
            win_frame["ModelProb"],
            win_frame["BestOdd"],
        )
    ]
    return win_frame


def build_historical_model_data(
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Carrega historico, cria features e retorna dataset modelavel."""
    raw_data = load_football_data(config.leagues, config.seasons, config.raw_dir)
    prepared_data, *_ = prepare_initial_data(raw_data)
    featured_data, feature_cols = add_rolling_features(
        prepared_data,
        window=config.rolling_window,
    )
    model_data = add_target_and_drop_na(featured_data, feature_cols)
    return prepared_data, model_data, feature_cols


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
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = config.output_dir / UPCOMING_PREDICTIONS_FILE
    odds_path = config.output_dir / UPCOMING_ODDS_FILE

    if fixtures.empty:
        pd.DataFrame().to_csv(predictions_path, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(odds_path, index=False, encoding="utf-8-sig")
        print("[fixtures] Nenhum jogo futuro encontrado para os filtros atuais.")
        return pd.DataFrame(), pd.DataFrame()

    bookmaker_odds = build_bookmaker_odds_long(fixtures)
    bookmaker_odds.to_csv(odds_path, index=False, encoding="utf-8-sig")

    prepared_data, model_data, feature_cols = build_historical_model_data(config)
    featured_fixtures = add_upcoming_rolling_features(
        prepared_data,
        fixtures,
        feature_cols,
        config.rolling_window,
    )

    if featured_fixtures.empty:
        pd.DataFrame().to_csv(predictions_path, index=False, encoding="utf-8-sig")
        print("[features] Nenhuma fixture com historico suficiente para prever.")
        return pd.DataFrame(), bookmaker_odds

    prediction_frames: list[pd.DataFrame] = []

    if "over25" in config.markets:
        over_data, over_feature_cols = prepare_market_dataset(
            model_data,
            feature_cols,
            TARGET_COL,
            OVER_MARKET_FEATURES,
        )
        if not over_data.empty:
            print("[modelo] Treinando modelo final Over 2.5...")
            over_model = train_calibrated_xgboost_model(
                over_data.loc[:, over_feature_cols],
                over_data[TARGET_COL],
                config.calibration_size,
                config.calibration_method,
            )
            prediction_frames.append(
                score_over25_predictions(
                    featured_fixtures,
                    over_model,
                    feature_cols,
                    config,
                )
            )
        else:
            print("[aviso] Over 2.5 sem dados/odds suficientes para prever.")

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
            )
            if "result" in config.markets:
                prediction_frames.append(
                    score_result_predictions(
                        featured_fixtures,
                        result_model,
                        feature_cols,
                        config,
                    )
                )
            if "win" in config.markets:
                prediction_frames.append(
                    score_win_predictions(
                        featured_fixtures,
                        result_model,
                        feature_cols,
                        config,
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
    print(f"[saida] Palpites futuros salvos em: {predictions_path}")
    print(f"[saida] Odds por casa salvas em: {odds_path}")

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
        choices=["over25", "result", "win", "all"],
        default=["all"],
    )
    parser.add_argument("--days-ahead", type=int, default=7)
    parser.add_argument("--raw-dir", default="raw_data")
    parser.add_argument("--output-dir", default="outputs")
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
    parser.add_argument("--min-result-prob", type=float, default=0.48)
    parser.add_argument("--max-result-odd", type=float, default=2.50)
    parser.add_argument("--min-win-prob", type=float, default=0.50)
    parser.add_argument("--max-win-odd", type=float, default=2.50)
    parser.add_argument(
        "--force-refresh-fixtures",
        action="store_true",
        help="Forca novo download do CSV de fixtures.",
    )

    args = parser.parse_args()
    markets = ["over25", "result", "win"] if "all" in args.markets else args.markets
    max_over_odd = args.max_over_odd if args.max_over_odd > 0 else None
    max_result_odd = args.max_result_odd if args.max_result_odd > 0 else None
    max_win_odd = args.max_win_odd if args.max_win_odd > 0 else None

    config = PipelineConfig(
        leagues=args.leagues,
        seasons=args.seasons,
        markets=markets,
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        rolling_window=args.rolling_window,
        calibration_size=args.calibration_size,
        calibration_method=args.calibration_method,
        walk_forward_splits=0,
        stake=args.stake,
        edge=args.edge,
        min_model_prob=args.min_model_prob,
        max_over_odd=max_over_odd,
        min_result_prob=args.min_result_prob,
        max_result_odd=max_result_odd,
        min_win_prob=args.min_win_prob,
        max_win_odd=max_win_odd,
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
