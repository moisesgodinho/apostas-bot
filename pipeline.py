"""Orquestracao da esteira completa de modelagem e backtests."""

from __future__ import annotations

from config import (
    NO_VIG_AWAY_COL,
    NO_VIG_DRAW_COL,
    NO_VIG_HOME_COL,
    NO_VIG_OVER_COL,
    ODD_AWAY_COL,
    ODD_DRAW_COL,
    ODD_HOME_COL,
    ODD_OVER_COL,
    ODD_UNDER_COL,
    OVERROUND_1X2_COL,
    OVERROUND_COL,
    PipelineConfig,
    RAW_IMPLIED_AWAY_COL,
    RAW_IMPLIED_DRAW_COL,
    RAW_IMPLIED_HOME_COL,
    RAW_IMPLIED_OVER_COL,
    RAW_IMPLIED_UNDER_COL,
    RESULT_TARGET_COL,
    TARGET_COL,
)
from backtesting import (
    print_match_result_results,
    print_results,
    print_win_results,
    run_backtest,
    run_match_result_backtest,
    run_match_result_walk_forward_validation,
    run_walk_forward_validation,
    run_win_backtest,
    run_win_walk_forward_validation,
)
from data_pipeline import (
    add_rolling_features,
    add_target_and_drop_na,
    chronological_train_test_split,
    load_football_data,
    prepare_initial_data,
)
from modeling import (
    evaluate_match_result_model,
    evaluate_model,
    train_calibrated_xgboost_model,
)


OVER_MARKET_FEATURES = [
    ODD_OVER_COL,
    ODD_UNDER_COL,
    RAW_IMPLIED_OVER_COL,
    RAW_IMPLIED_UNDER_COL,
    NO_VIG_OVER_COL,
    OVERROUND_COL,
]
RESULT_MARKET_FEATURES = [
    ODD_HOME_COL,
    ODD_DRAW_COL,
    ODD_AWAY_COL,
    RAW_IMPLIED_HOME_COL,
    RAW_IMPLIED_DRAW_COL,
    RAW_IMPLIED_AWAY_COL,
    NO_VIG_HOME_COL,
    NO_VIG_DRAW_COL,
    NO_VIG_AWAY_COL,
    OVERROUND_1X2_COL,
]


def prepare_market_dataset(
    data,
    feature_cols,
    target_col: str,
    market_feature_cols,
):
    """Remove linhas sem odds/features exigidas por um mercado especifico."""
    model_cols = list(feature_cols) + list(market_feature_cols)
    market_data = data.dropna(subset=model_cols + [target_col]).copy()
    for col in market_feature_cols:
        if col.startswith("Odd_"):
            market_data = market_data[market_data[col] > 1.0]
    market_data = market_data.sort_values("MatchDatetime", kind="mergesort")
    return market_data.reset_index(drop=True), model_cols


def run_pipeline(config: PipelineConfig) -> None:
    """Executa a esteira completa de ponta a ponta."""
    raw_data = load_football_data(config.leagues, config.seasons, config.raw_dir)
    prepared_data, *_ = prepare_initial_data(raw_data)
    featured_data, feature_cols = add_rolling_features(
        prepared_data,
        window=config.rolling_window,
    )
    model_data = add_target_and_drop_na(featured_data, feature_cols)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if "over25" in config.markets:
        over_data, over_feature_cols = prepare_market_dataset(
            model_data,
            feature_cols,
            TARGET_COL,
            OVER_MARKET_FEATURES,
        )
        if over_data.empty:
            print("[aviso] Mercado Over 2.5 ignorado: sem odds Over/Under validas.")
        else:
            x_train, x_test, y_train, y_test = chronological_train_test_split(
                over_data,
                over_feature_cols,
                config.train_size,
            )

            print("[modelo] Treinando XGBClassifier Over 2.5 calibrado...")
            model = train_calibrated_xgboost_model(
                x_train,
                y_train,
                config.calibration_size,
                config.calibration_method,
            )
            probabilities = model.predict_proba(x_test)[:, 1]

            metrics = evaluate_model(y_test, probabilities)
            test_data = over_data.iloc[len(x_train) :].copy()
            backtest, backtest_summary = run_backtest(
                test_data,
                probabilities,
                stake=config.stake,
                edge=config.edge,
                min_model_prob=config.min_model_prob,
                max_over_odd=config.max_over_odd,
            )

            output_path = config.output_dir / "backtest_over25_results.csv"
            backtest.to_csv(output_path, index=False, encoding="utf-8-sig")

            print_results(
                metrics,
                backtest_summary,
                config.stake,
                config.edge,
                config.min_model_prob,
                config.max_over_odd,
            )
            print(f"\n[saida] Backtest Over 2.5 salvo em: {output_path}")

            walk_forward_summary = run_walk_forward_validation(
                over_data,
                over_feature_cols,
                config,
            )
            if not walk_forward_summary.empty:
                walk_forward_path = (
                    config.output_dir / "walk_forward_over25_summary.csv"
                )
                walk_forward_summary.to_csv(
                    walk_forward_path,
                    index=False,
                    encoding="utf-8-sig",
                )
                print(f"[saida] Walk-forward Over 2.5 salvo em: {walk_forward_path}")

    if "result" in config.markets or "win" in config.markets:
        result_data, result_feature_cols = prepare_market_dataset(
            model_data,
            feature_cols,
            RESULT_TARGET_COL,
            RESULT_MARKET_FEATURES,
        )
        if result_data.empty:
            print("[aviso] Mercados 1X2/Vitoria ignorados: sem odds 1X2 validas.")
            return

        x_train, x_test, y_train, y_test = chronological_train_test_split(
            result_data,
            result_feature_cols,
            config.train_size,
            target_col=RESULT_TARGET_COL,
        )

        print("[modelo] Treinando XGBClassifier Resultado Final 1X2 calibrado...")
        result_model = train_calibrated_xgboost_model(
            x_train,
            y_train,
            config.calibration_size,
            config.calibration_method,
        )
        result_probabilities = result_model.predict_proba(x_test)

        result_metrics = evaluate_match_result_model(y_test, result_probabilities)
        test_data = result_data.iloc[len(x_train) :].copy()

        if "result" in config.markets:
            result_backtest, result_backtest_summary = run_match_result_backtest(
                test_data,
                result_probabilities,
                stake=config.stake,
                edge=config.edge,
                min_model_prob=config.min_result_prob,
                max_result_odd=config.max_result_odd,
            )

            result_output_path = config.output_dir / "backtest_result_1x2_results.csv"
            result_backtest.to_csv(
                result_output_path,
                index=False,
                encoding="utf-8-sig",
            )

            print_match_result_results(
                result_metrics,
                result_backtest_summary,
                config.stake,
                config.edge,
                config.min_result_prob,
                config.max_result_odd,
            )
            print(f"\n[saida] Backtest 1X2 salvo em: {result_output_path}")

            result_walk_forward_summary = run_match_result_walk_forward_validation(
                result_data,
                result_feature_cols,
                config,
            )
            if not result_walk_forward_summary.empty:
                result_walk_forward_path = (
                    config.output_dir / "walk_forward_result_1x2_summary.csv"
                )
                result_walk_forward_summary.to_csv(
                    result_walk_forward_path,
                    index=False,
                    encoding="utf-8-sig",
                )
                print(
                    "[saida] Walk-forward 1X2 salvo em: "
                    f"{result_walk_forward_path}"
                )

        if "win" in config.markets:
            win_backtest, win_backtest_summary = run_win_backtest(
                test_data,
                result_probabilities,
                stake=config.stake,
                edge=config.edge,
                min_model_prob=config.min_win_prob,
                max_win_odd=config.max_win_odd,
            )

            win_output_path = config.output_dir / "backtest_win_results.csv"
            win_backtest.to_csv(
                win_output_path,
                index=False,
                encoding="utf-8-sig",
            )

            print_win_results(
                result_metrics,
                win_backtest_summary,
                config.stake,
                config.edge,
                config.min_win_prob,
                config.max_win_odd,
            )
            print(f"\n[saida] Backtest Vitoria salvo em: {win_output_path}")

            win_walk_forward_summary = run_win_walk_forward_validation(
                result_data,
                result_feature_cols,
                config,
            )
            if not win_walk_forward_summary.empty:
                win_walk_forward_path = (
                    config.output_dir / "walk_forward_win_summary.csv"
                )
                win_walk_forward_summary.to_csv(
                    win_walk_forward_path,
                    index=False,
                    encoding="utf-8-sig",
                )
                print(
                    "[saida] Walk-forward Vitoria salvo em: "
                    f"{win_walk_forward_path}"
                )
