"""Comparacao entre modelos com odds e sem odds."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from backtesting import (
    run_backtest,
    run_match_result_backtest,
    run_under25_backtest,
    run_win_backtest,
)
from config import PipelineConfig, RESULT_TARGET_COL, TARGET_COL
from data_pipeline import chronological_train_test_split
from modeling import (
    evaluate_match_result_model,
    evaluate_model,
    train_calibrated_xgboost_model,
)


def _format_period(data: pd.DataFrame) -> tuple[str, str]:
    """Retorna inicio e fim ISO de um conjunto temporal."""
    return (
        data["MatchDatetime"].min().date().isoformat(),
        data["MatchDatetime"].max().date().isoformat(),
    )


def _build_variant_features(
    base_feature_cols: Sequence[str],
    market_feature_cols: Sequence[str],
    use_odds: bool,
) -> list[str]:
    """Monta features da variante com ou sem odds."""
    if use_odds:
        return list(base_feature_cols) + list(market_feature_cols)
    return list(base_feature_cols)


def _summary_fields(summary: dict[str, float]) -> dict[str, float]:
    """Padroniza campos financeiros do backtest."""
    return {
        "Bets": summary["bets"],
        "TotalStaked": summary["total_staked"],
        "TotalProfit": summary["total_profit"],
        "ROI": summary["roi"],
        "HitRate": summary["hit_rate"],
        "AvgEdge": summary["avg_edge"],
        "AvgModelProb": summary["avg_model_prob"],
        "AvgOdd": summary["avg_odd"],
    }


def _empty_metric_fields() -> dict[str, float]:
    """Cria metricas vazias para manter o CSV retangular."""
    return {
        "Accuracy": np.nan,
        "PrecisionOver": np.nan,
        "PrecisionHome": np.nan,
        "PrecisionDraw": np.nan,
        "PrecisionAway": np.nan,
        "LogLoss": np.nan,
        "BrierScore": np.nan,
        "CalibrationECE": np.nan,
    }


def _binary_metric_fields(metrics: dict[str, float]) -> dict[str, float]:
    """Mapeia metricas binarias para o formato da comparacao."""
    fields = _empty_metric_fields()
    fields.update(
        {
            "Accuracy": metrics["accuracy"],
            "PrecisionOver": metrics["precision_over"],
            "LogLoss": metrics["logloss"],
            "BrierScore": metrics["brier_score"],
            "CalibrationECE": metrics["calibration_ece"],
        }
    )
    return fields


def _result_metric_fields(metrics: dict[str, float]) -> dict[str, float]:
    """Mapeia metricas 1X2 para o formato da comparacao."""
    fields = _empty_metric_fields()
    fields.update(
        {
            "Accuracy": metrics["accuracy"],
            "PrecisionHome": metrics["precision_h"],
            "PrecisionDraw": metrics.get("precision_d", np.nan),
            "PrecisionAway": metrics["precision_a"],
            "LogLoss": metrics["logloss"],
            "BrierScore": metrics["brier_score"],
            "CalibrationECE": metrics["calibration_ece"],
        }
    )
    return fields


def _comparison_row(
    market: str,
    variant: str,
    uses_odds: bool,
    feature_count: int,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    metrics: dict[str, float],
    summary: dict[str, float],
) -> dict[str, float | int | str | bool]:
    """Cria uma linha padronizada de comparacao."""
    train_start, train_end = _format_period(train_data)
    test_start, test_end = _format_period(test_data)

    return {
        "Market": market,
        "ModelVariant": variant,
        "UsesOdds": uses_odds,
        "FeatureCount": feature_count,
        "TrainRows": len(train_data),
        "TestRows": len(test_data),
        "TrainStart": train_start,
        "TrainEnd": train_end,
        "TestStart": test_start,
        "TestEnd": test_end,
        **metrics,
        **_summary_fields(summary),
    }


def _run_variant(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    config: PipelineConfig,
) -> tuple[object, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Treina uma variante preservando a divisao temporal."""
    x_train, x_test, y_train, y_test = chronological_train_test_split(
        data,
        feature_cols,
        config.train_size,
        target_col=target_col,
    )
    model = train_calibrated_xgboost_model(
        x_train,
        y_train,
        config.calibration_size,
        config.calibration_method,
        config.xgb_tuning_trials,
        config.xgb_tuning_validation_size,
    )
    train_data = data.iloc[: len(x_train)].copy()
    test_data = data.iloc[len(x_train) :].copy()
    return model, train_data, test_data, y_test, x_test


def run_over25_model_comparison(
    data: pd.DataFrame,
    base_feature_cols: Sequence[str],
    market_feature_cols: Sequence[str],
    config: PipelineConfig,
    include_over: bool = True,
    include_under: bool = False,
) -> pd.DataFrame:
    """Compara mercados Over/Under 2.5 com e sem odds nas features."""
    rows = []
    print("\n========== COMPARACAO DE MODELOS: OVER/UNDER 2.5 ==========")

    for variant, uses_odds in [("Sem odds", False), ("Com odds", True)]:
        feature_cols = _build_variant_features(
            base_feature_cols,
            market_feature_cols,
            uses_odds,
        )
        print(f"[comparacao] Treinando Over 2.5 - {variant}")
        model, train_data, test_data, y_test, x_test = _run_variant(
            data,
            feature_cols,
            TARGET_COL,
            config,
        )
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics = _binary_metric_fields(evaluate_model(y_test, probabilities))

        if include_over:
            _, backtest_summary = run_backtest(
                test_data,
                probabilities,
                stake=config.stake,
                edge=config.edge,
                min_model_prob=config.min_model_prob,
                max_over_odd=config.max_over_odd,
            )
            rows.append(
                _comparison_row(
                    "Over 2.5",
                    variant,
                    uses_odds,
                    len(feature_cols),
                    train_data,
                    test_data,
                    metrics,
                    backtest_summary,
                )
            )

        if include_under:
            _, under_summary = run_under25_backtest(
                test_data,
                probabilities,
                stake=config.stake,
                edge=config.edge,
                min_model_prob=config.min_under_prob,
                max_under_odd=config.max_under_odd,
            )
            rows.append(
                _comparison_row(
                    "Under 2.5",
                    variant,
                    uses_odds,
                    len(feature_cols),
                    train_data,
                    test_data,
                    metrics,
                    under_summary,
                )
            )

    return pd.DataFrame(rows)


def run_result_family_model_comparison(
    data: pd.DataFrame,
    base_feature_cols: Sequence[str],
    market_feature_cols: Sequence[str],
    config: PipelineConfig,
    include_result: bool,
    include_win: bool,
) -> pd.DataFrame:
    """Compara modelos 1X2/Vitoria com e sem odds nas features."""
    rows = []
    print("\n========== COMPARACAO DE MODELOS: 1X2 / VITORIA ==========")

    backtest_jobs: list[
        tuple[
            str,
            Callable[[pd.DataFrame, np.ndarray], tuple[pd.DataFrame, dict[str, float]]],
        ]
    ] = []
    if include_result:
        backtest_jobs.append(
            (
                "Resultado 1X2",
                lambda test_data, probabilities: run_match_result_backtest(
                    test_data,
                    probabilities,
                    stake=config.stake,
                    edge=config.edge,
                    min_model_prob=config.min_result_prob,
                    max_result_odd=config.max_result_odd,
                ),
            )
        )
    if include_win:
        backtest_jobs.append(
            (
                "Vitoria Casa/Fora",
                lambda test_data, probabilities: run_win_backtest(
                    test_data,
                    probabilities,
                    stake=config.stake,
                    edge=config.edge,
                    min_model_prob=config.min_win_prob,
                    max_win_odd=config.max_win_odd,
                ),
            )
        )

    for variant, uses_odds in [("Sem odds", False), ("Com odds", True)]:
        feature_cols = _build_variant_features(
            base_feature_cols,
            market_feature_cols,
            uses_odds,
        )
        print(f"[comparacao] Treinando 1X2/Vitoria - {variant}")
        model, train_data, test_data, y_test, x_test = _run_variant(
            data,
            feature_cols,
            RESULT_TARGET_COL,
            config,
        )
        probabilities = model.predict_proba(x_test)
        metrics = _result_metric_fields(
            evaluate_match_result_model(y_test, probabilities)
        )

        for market, backtest_func in backtest_jobs:
            _, backtest_summary = backtest_func(test_data, probabilities)
            rows.append(
                _comparison_row(
                    market,
                    variant,
                    uses_odds,
                    len(feature_cols),
                    train_data,
                    test_data,
                    metrics,
                    backtest_summary,
                )
            )

    return pd.DataFrame(rows)


def print_model_comparison_summary(comparison: pd.DataFrame) -> None:
    """Imprime uma visao compacta da comparacao."""
    if comparison.empty:
        return

    print("\n========== RESUMO COMPARACAO COM ODDS VS SEM ODDS ==========")
    display_cols = [
        "Market",
        "ModelVariant",
        "Accuracy",
        "LogLoss",
        "BrierScore",
        "CalibrationECE",
        "Bets",
        "ROI",
        "HitRate",
        "TotalProfit",
    ]
    printable = comparison[display_cols].copy()
    for _, row in printable.iterrows():
        print(
            f"{row['Market']} | {row['ModelVariant']}: "
            f"Acc {row['Accuracy']:.2%}, "
            f"LogLoss {row['LogLoss']:.4f}, "
            f"Brier {row['BrierScore']:.4f}, "
            f"ECE {row['CalibrationECE']:.2%}, "
            f"Apostas {row['Bets']:.0f}, "
            f"ROI {row['ROI']:.2%}, "
            f"Acerto {row['HitRate']:.2%}, "
            f"Lucro R$ {row['TotalProfit']:.2f}"
        )
