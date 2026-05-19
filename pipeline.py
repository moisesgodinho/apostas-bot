"""Orquestracao da esteira completa de modelagem e backtests."""

from __future__ import annotations

import pandas as pd

from config import (
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
    print_under25_results,
    print_win_results,
    run_backtest,
    run_match_result_backtest,
    run_match_result_walk_forward_validation,
    run_under25_backtest,
    run_walk_forward_validation,
    run_win_backtest,
    run_win_walk_forward_validation,
)
from data_pipeline import (
    add_elo_features,
    add_rolling_features,
    add_target_and_drop_na,
    load_football_data,
    prepare_initial_data,
)
from filter_optimizer import (
    optimize_market_filters,
    print_filter_optimization_summary,
)
from modeling import (
    blend_probabilities,
    build_calibration_curve_frame,
    evaluate_match_result_model,
    evaluate_model,
    summarize_calibration_curve,
    tune_probability_blend,
    train_calibrated_xgboost_model,
)
from model_comparison import (
    print_model_comparison_summary,
    run_over25_model_comparison,
    run_result_family_model_comparison,
)
from realistic_backtesting import (
    SeasonSplit,
    run_realistic_market_backtest,
    split_train_validation_test_by_season,
)
from understat_xg import merge_understat_xg_features


OVER_MARKET_FEATURES = [
    ODD_OVER_COL,
    ODD_UNDER_COL,
    RAW_IMPLIED_OVER_COL,
    RAW_IMPLIED_UNDER_COL,
    NO_VIG_OVER_COL,
    NO_VIG_UNDER_COL,
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


def split_market_model_data(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: PipelineConfig,
) -> SeasonSplit:
    """Separa dados de modelagem em treino, validacao e teste."""
    if config.split_strategy == "season":
        return split_train_validation_test_by_season(
            data,
            feature_cols,
            target_col,
            validation_seasons=config.validation_seasons,
            test_seasons=config.test_seasons,
        )

    ordered = data.sort_values("MatchDatetime", kind="mergesort").reset_index(
        drop=True
    )
    train_end = int(len(ordered) * config.train_size)
    validation_end = train_end + max(1, (len(ordered) - train_end) // 2)
    if train_end <= 0 or validation_end >= len(ordered):
        raise ValueError("Dados insuficientes para split cronologico 3-way.")

    train_data = ordered.iloc[:train_end].copy()
    validation_data = ordered.iloc[train_end:validation_end].copy()
    test_data = ordered.iloc[validation_end:].copy()
    print(
        "[split cronologico] Treino/validacao/teste: "
        f"{len(train_data):,}/{len(validation_data):,}/{len(test_data):,} jogos"
    )
    return SeasonSplit(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        x_train=train_data.loc[:, feature_cols],
        x_validation=validation_data.loc[:, feature_cols],
        x_test=test_data.loc[:, feature_cols],
        y_train=train_data[target_col],
        y_validation=validation_data[target_col],
        y_test=test_data[target_col],
        train_seasons=sorted(train_data["Temporada"].astype(str).unique().tolist()),
        validation_seasons=sorted(
            validation_data["Temporada"].astype(str).unique().tolist()
        ),
        test_seasons=sorted(test_data["Temporada"].astype(str).unique().tolist()),
    )


def build_model_artifact_frames(
    model,
    market: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrai importancia de features e resumo de tuning de um modelo."""
    importances = getattr(model, "base_feature_importances_", None)
    if importances is None:
        feature_importance = pd.DataFrame()
    else:
        feature_importance = pd.DataFrame(
            {
                "Market": market,
                "Feature": feature_cols,
                "Importance": importances,
            }
        ).sort_values("Importance", ascending=False, kind="mergesort")
        total_importance = float(feature_importance["Importance"].sum())
        if total_importance > 0:
            feature_importance["ImportanceShare"] = (
                feature_importance["Importance"] / total_importance
            )
        else:
            feature_importance["ImportanceShare"] = 0.0
        feature_importance["Rank"] = range(1, len(feature_importance) + 1)

    tuning_summary = getattr(model, "xgb_tuning_summary_", pd.DataFrame())
    if not tuning_summary.empty:
        tuning_summary = tuning_summary.copy()
        tuning_summary.insert(0, "Market", market)

    return feature_importance, tuning_summary


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
    prepared_data = merge_understat_xg_features(
        prepared_data,
        cache_dir=config.understat_xg_dir,
        enabled=config.use_understat_xg,
        force_refresh=config.force_refresh_understat_xg,
    )
    prepared_data = add_elo_features(
        prepared_data,
        initial_rating=config.elo_initial,
        k_factor=config.elo_k_factor,
        home_advantage=config.elo_home_advantage,
    )
    featured_data, feature_cols = add_rolling_features(
        prepared_data,
        window=config.rolling_window,
        feature_profile=config.feature_profile,
        lineup_features_path=config.lineup_features_path,
    )
    model_data = add_target_and_drop_na(featured_data, feature_cols)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_frames = []
    optimization_frames = []
    optimization_grid_frames = []
    feature_importance_frames = []
    tuning_summary_frames = []
    probability_blend_frames = []
    calibration_curve_frames = []
    realistic_summary_frames = []
    realistic_grid_frames = []
    realistic_bet_frames = []
    realistic_monthly_frames = []
    realistic_league_frames = []

    totals_markets = {"over25", "under25"}.intersection(config.markets)
    if totals_markets:
        over_data, over_feature_cols = prepare_market_dataset(
            model_data,
            feature_cols,
            TARGET_COL,
            OVER_MARKET_FEATURES,
        )
        if over_data.empty:
            print("[aviso] Mercados Over/Under 2.5 ignorados: sem odds validas.")
        else:
            split = split_market_model_data(
                over_data,
                over_feature_cols,
                TARGET_COL,
                config,
            )

            print("[modelo] Treinando XGBClassifier Over 2.5 calibrado...")
            model = train_calibrated_xgboost_model(
                split.x_train,
                split.y_train,
                config.calibration_size,
                config.calibration_method,
                config.xgb_tuning_trials,
                config.xgb_tuning_validation_size,
            )
            feature_importance, tuning_summary = build_model_artifact_frames(
                model,
                "Total Gols 2.5",
                over_feature_cols,
            )
            feature_importance_frames.append(feature_importance)
            tuning_summary_frames.append(tuning_summary)
            validation_probabilities = model.predict_proba(split.x_validation)[:, 1]
            probabilities = model.predict_proba(split.x_test)[:, 1]
            validation_data = split.validation_data.copy()
            test_data = split.test_data.copy()
            validation_market_probabilities = validation_data[
                NO_VIG_OVER_COL
            ].to_numpy(dtype=float)
            test_market_probabilities = test_data[NO_VIG_OVER_COL].to_numpy(
                dtype=float
            )
            blend_alpha, blend_summary = tune_probability_blend(
                split.y_validation,
                validation_probabilities,
                validation_market_probabilities,
                labels=[0, 1],
                market="Total Gols 2.5",
            )
            probability_blend_frames.append(blend_summary)
            validation_probabilities = blend_probabilities(
                validation_probabilities,
                validation_market_probabilities,
                blend_alpha,
            )
            probabilities = blend_probabilities(
                probabilities,
                test_market_probabilities,
                blend_alpha,
            )
            metrics = evaluate_model(split.y_test, probabilities)

            if "over25" in config.markets:
                calibration_curve_frames.append(
                    build_calibration_curve_frame(
                        split.y_test,
                        probabilities,
                        "Over 2.5",
                        "Over 2.5",
                        positive_label=1,
                    )
                )
                validation_backtest, _ = run_backtest(
                    validation_data,
                    validation_probabilities,
                    stake=config.stake,
                    edge=config.edge,
                    min_model_prob=config.min_model_prob,
                    max_over_odd=config.max_over_odd,
                )
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

                if config.run_filter_optimization:
                    best_filters, filter_grid = optimize_market_filters(
                        backtest,
                        "Over 2.5",
                        config,
                    )
                    optimization_frames.append(best_filters)
                    optimization_grid_frames.append(filter_grid)

                if config.run_realistic_backtest:
                    (
                        realistic_summary,
                        realistic_grid,
                        realistic_bets,
                        realistic_monthly,
                        realistic_league,
                    ) = run_realistic_market_backtest(
                        validation_backtest,
                        backtest,
                        "Over 2.5",
                        config,
                    )
                    realistic_summary_frames.append(realistic_summary)
                    realistic_grid_frames.append(realistic_grid)
                    realistic_bet_frames.append(realistic_bets)
                    realistic_monthly_frames.append(realistic_monthly)
                    realistic_league_frames.append(realistic_league)

                print_results(
                    metrics,
                    backtest_summary,
                    config.stake,
                    config.edge,
                    config.min_model_prob,
                    config.max_over_odd,
                )
                print(f"\n[saida] Backtest Over 2.5 salvo em: {output_path}")

            if "under25" in config.markets:
                calibration_curve_frames.append(
                    build_calibration_curve_frame(
                        split.y_test,
                        1.0 - probabilities,
                        "Under 2.5",
                        "Under 2.5",
                        positive_label=0,
                    )
                )
                validation_under_backtest, _ = run_under25_backtest(
                    validation_data,
                    validation_probabilities,
                    stake=config.stake,
                    edge=config.edge,
                    min_model_prob=config.min_under_prob,
                    max_under_odd=config.max_under_odd,
                )
                under_backtest, under_backtest_summary = run_under25_backtest(
                    test_data,
                    probabilities,
                    stake=config.stake,
                    edge=config.edge,
                    min_model_prob=config.min_under_prob,
                    max_under_odd=config.max_under_odd,
                )

                under_output_path = config.output_dir / "backtest_under25_results.csv"
                under_backtest.to_csv(
                    under_output_path,
                    index=False,
                    encoding="utf-8-sig",
                )

                if config.run_filter_optimization:
                    best_filters, filter_grid = optimize_market_filters(
                        under_backtest,
                        "Under 2.5",
                        config,
                    )
                    optimization_frames.append(best_filters)
                    optimization_grid_frames.append(filter_grid)

                if config.run_realistic_backtest:
                    (
                        realistic_summary,
                        realistic_grid,
                        realistic_bets,
                        realistic_monthly,
                        realistic_league,
                    ) = run_realistic_market_backtest(
                        validation_under_backtest,
                        under_backtest,
                        "Under 2.5",
                        config,
                    )
                    realistic_summary_frames.append(realistic_summary)
                    realistic_grid_frames.append(realistic_grid)
                    realistic_bet_frames.append(realistic_bets)
                    realistic_monthly_frames.append(realistic_monthly)
                    realistic_league_frames.append(realistic_league)

                print_under25_results(
                    metrics,
                    under_backtest_summary,
                    config.stake,
                    config.edge,
                    config.min_under_prob,
                    config.max_under_odd,
                )
                print(f"\n[saida] Backtest Under 2.5 salvo em: {under_output_path}")

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

            if config.run_model_comparison:
                comparison_frames.append(
                    run_over25_model_comparison(
                        over_data,
                        feature_cols,
                        OVER_MARKET_FEATURES,
                        config,
                        include_over="over25" in config.markets,
                        include_under="under25" in config.markets,
                    )
                )

    if "result" in config.markets or "win" in config.markets:
        result_data, result_feature_cols = prepare_market_dataset(
            model_data,
            feature_cols,
            RESULT_TARGET_COL,
            RESULT_MARKET_FEATURES,
        )
        if result_data.empty:
            print("[aviso] Mercados 1X2/Vitoria ignorados: sem odds 1X2 validas.")
        else:
            result_split = split_market_model_data(
                result_data,
                result_feature_cols,
                RESULT_TARGET_COL,
                config,
            )

            print("[modelo] Treinando XGBClassifier Resultado Final 1X2 calibrado...")
            result_model = train_calibrated_xgboost_model(
                result_split.x_train,
                result_split.y_train,
                config.calibration_size,
                config.calibration_method,
                config.xgb_tuning_trials,
                config.xgb_tuning_validation_size,
            )
            feature_importance, tuning_summary = build_model_artifact_frames(
                result_model,
                "Resultado 1X2 / Vitoria",
                result_feature_cols,
            )
            feature_importance_frames.append(feature_importance)
            tuning_summary_frames.append(tuning_summary)
            result_validation_probabilities = result_model.predict_proba(
                result_split.x_validation
            )
            result_probabilities = result_model.predict_proba(result_split.x_test)
            validation_data = result_split.validation_data.copy()
            test_data = result_split.test_data.copy()
            validation_market_probabilities = validation_data[
                [NO_VIG_HOME_COL, NO_VIG_DRAW_COL, NO_VIG_AWAY_COL]
            ].to_numpy(dtype=float)
            test_market_probabilities = test_data[
                [NO_VIG_HOME_COL, NO_VIG_DRAW_COL, NO_VIG_AWAY_COL]
            ].to_numpy(dtype=float)
            blend_alpha, blend_summary = tune_probability_blend(
                result_split.y_validation,
                result_validation_probabilities,
                validation_market_probabilities,
                labels=[0, 1, 2],
                market="Resultado 1X2 / Vitoria",
            )
            probability_blend_frames.append(blend_summary)
            result_validation_probabilities = blend_probabilities(
                result_validation_probabilities,
                validation_market_probabilities,
                blend_alpha,
            )
            result_probabilities = blend_probabilities(
                result_probabilities,
                test_market_probabilities,
                blend_alpha,
            )
            result_metrics = evaluate_match_result_model(
                result_split.y_test,
                result_probabilities,
            )

            if "result" in config.markets:
                for class_index, outcome in enumerate(["Casa", "Empate", "Fora"]):
                    calibration_curve_frames.append(
                        build_calibration_curve_frame(
                            result_split.y_test,
                            result_probabilities[:, class_index],
                            "Resultado 1X2",
                            outcome,
                            positive_label=class_index,
                        )
                    )
                result_validation_backtest, _ = run_match_result_backtest(
                    validation_data,
                    result_validation_probabilities,
                    stake=config.stake,
                    edge=config.edge,
                    min_model_prob=config.min_result_prob,
                    max_result_odd=config.max_result_odd,
                )
                result_backtest, result_backtest_summary = run_match_result_backtest(
                    test_data,
                    result_probabilities,
                    stake=config.stake,
                    edge=config.edge,
                    min_model_prob=config.min_result_prob,
                    max_result_odd=config.max_result_odd,
                )

                result_output_path = (
                    config.output_dir / "backtest_result_1x2_results.csv"
                )
                result_backtest.to_csv(
                    result_output_path,
                    index=False,
                    encoding="utf-8-sig",
                )

                if config.run_filter_optimization:
                    best_filters, filter_grid = optimize_market_filters(
                        result_backtest,
                        "Resultado 1X2",
                        config,
                    )
                    optimization_frames.append(best_filters)
                    optimization_grid_frames.append(filter_grid)

                if config.run_realistic_backtest:
                    (
                        realistic_summary,
                        realistic_grid,
                        realistic_bets,
                        realistic_monthly,
                        realistic_league,
                    ) = run_realistic_market_backtest(
                        result_validation_backtest,
                        result_backtest,
                        "Resultado 1X2",
                        config,
                    )
                    realistic_summary_frames.append(realistic_summary)
                    realistic_grid_frames.append(realistic_grid)
                    realistic_bet_frames.append(realistic_bets)
                    realistic_monthly_frames.append(realistic_monthly)
                    realistic_league_frames.append(realistic_league)

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
                for class_index, outcome in [(0, "Casa"), (2, "Fora")]:
                    calibration_curve_frames.append(
                        build_calibration_curve_frame(
                            result_split.y_test,
                            result_probabilities[:, class_index],
                            "Vitoria Casa/Fora",
                            outcome,
                            positive_label=class_index,
                        )
                    )
                win_validation_backtest, _ = run_win_backtest(
                    validation_data,
                    result_validation_probabilities,
                    stake=config.stake,
                    edge=config.edge,
                    min_model_prob=config.min_win_prob,
                    max_win_odd=config.max_win_odd,
                )
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

                if config.run_filter_optimization:
                    best_filters, filter_grid = optimize_market_filters(
                        win_backtest,
                        "Vitoria Casa/Fora",
                        config,
                    )
                    optimization_frames.append(best_filters)
                    optimization_grid_frames.append(filter_grid)

                if config.run_realistic_backtest:
                    (
                        realistic_summary,
                        realistic_grid,
                        realistic_bets,
                        realistic_monthly,
                        realistic_league,
                    ) = run_realistic_market_backtest(
                        win_validation_backtest,
                        win_backtest,
                        "Vitoria Casa/Fora",
                        config,
                    )
                    realistic_summary_frames.append(realistic_summary)
                    realistic_grid_frames.append(realistic_grid)
                    realistic_bet_frames.append(realistic_bets)
                    realistic_monthly_frames.append(realistic_monthly)
                    realistic_league_frames.append(realistic_league)

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

            if config.run_model_comparison:
                comparison_frames.append(
                    run_result_family_model_comparison(
                        result_data,
                        feature_cols,
                        RESULT_MARKET_FEATURES,
                        config,
                        include_result="result" in config.markets,
                        include_win="win" in config.markets,
                    )
                )

    comparison_frames = [frame for frame in comparison_frames if not frame.empty]
    if comparison_frames:
        comparison = pd.concat(
            comparison_frames,
            ignore_index=True,
        )
        comparison_path = config.output_dir / "model_comparison_summary.csv"
        comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        print_model_comparison_summary(comparison)
        print(f"[saida] Comparacao de modelos salva em: {comparison_path}")

    optimization_frames = [frame for frame in optimization_frames if not frame.empty]
    optimization_grid_frames = [
        frame for frame in optimization_grid_frames if not frame.empty
    ]
    if optimization_frames:
        optimization = pd.concat(optimization_frames, ignore_index=True)
        optimization_path = config.output_dir / "filter_optimization_summary.csv"
        optimization.to_csv(optimization_path, index=False, encoding="utf-8-sig")
        print_filter_optimization_summary(optimization)
        print(f"[saida] Otimizacao de filtros salva em: {optimization_path}")

    if optimization_grid_frames:
        optimization_grid = pd.concat(
            optimization_grid_frames,
            ignore_index=True,
        )
        optimization_grid_path = config.output_dir / "filter_optimization_grid.csv"
        optimization_grid.to_csv(
            optimization_grid_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[saida] Grade de filtros salva em: {optimization_grid_path}")

    realistic_outputs = [
        (
            realistic_summary_frames,
            "realistic_backtest_summary.csv",
            "Resumo de backtest realista",
        ),
        (
            realistic_grid_frames,
            "realistic_backtest_grid.csv",
            "Grade de backtest realista",
        ),
        (
            realistic_bet_frames,
            "realistic_backtest_bets.csv",
            "Apostas do backtest realista",
        ),
        (
            realistic_monthly_frames,
            "realistic_backtest_monthly.csv",
            "Lucro mensal do backtest realista",
        ),
        (
            realistic_league_frames,
            "realistic_backtest_league.csv",
            "Lucro por liga do backtest realista",
        ),
    ]
    for frames, file_name, label in realistic_outputs:
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            continue
        output = pd.concat(frames, ignore_index=True)
        output_path = config.output_dir / file_name
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[saida] {label} salvo em: {output_path}")

    feature_importance_frames = [
        frame for frame in feature_importance_frames if not frame.empty
    ]
    if feature_importance_frames:
        feature_importance = pd.concat(
            feature_importance_frames,
            ignore_index=True,
        )
        feature_importance_path = config.output_dir / "feature_importance_summary.csv"
        feature_importance.to_csv(
            feature_importance_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[saida] Importancia de features salva em: {feature_importance_path}")

    tuning_summary_frames = [
        frame for frame in tuning_summary_frames if not frame.empty
    ]
    if tuning_summary_frames:
        tuning_summary = pd.concat(tuning_summary_frames, ignore_index=True)
        tuning_summary_path = config.output_dir / "xgb_tuning_summary.csv"
        tuning_summary.to_csv(
            tuning_summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[saida] Tuning XGBoost salvo em: {tuning_summary_path}")

    probability_blend_frames = [
        frame for frame in probability_blend_frames if not frame.empty
    ]
    if probability_blend_frames:
        probability_blend = pd.concat(
            probability_blend_frames,
            ignore_index=True,
        )
        probability_blend_path = config.output_dir / "probability_blend_summary.csv"
        probability_blend.to_csv(
            probability_blend_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[saida] Blend de probabilidades salvo em: {probability_blend_path}")

    calibration_curve_frames = [
        frame for frame in calibration_curve_frames if not frame.empty
    ]
    if calibration_curve_frames:
        calibration_curve = pd.concat(
            calibration_curve_frames,
            ignore_index=True,
        )
        calibration_curve["CalibrationMethod"] = config.calibration_method
        calibration_curve_path = config.output_dir / "calibration_curve_summary.csv"
        calibration_curve.to_csv(
            calibration_curve_path,
            index=False,
            encoding="utf-8-sig",
        )

        calibration_metrics = summarize_calibration_curve(calibration_curve)
        calibration_metrics_path = (
            config.output_dir / "calibration_metrics_summary.csv"
        )
        calibration_metrics.to_csv(
            calibration_metrics_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(
            "[saida] Curvas e metricas de calibracao salvas em: "
            f"{calibration_curve_path} | {calibration_metrics_path}"
        )
