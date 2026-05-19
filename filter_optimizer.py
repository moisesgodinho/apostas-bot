"""Otimizacao cronologica de filtros de apostas por mercado."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import PipelineConfig


MARKET_COLUMNS = {
    "Over 2.5": {
        "prob_col": "Model_Prob_Over25",
        "edge_col": "Edge",
        "odd_col": "Odd_Over25",
        "target_col": "Over25",
        "prediction_col": None,
        "positive_target": 1,
    },
    "Under 2.5": {
        "prob_col": "Model_Prob_Under25",
        "edge_col": "Under_Edge",
        "odd_col": "Odd_Under25",
        "target_col": "Over25",
        "prediction_col": None,
        "positive_target": 0,
    },
    "Resultado 1X2": {
        "prob_col": "Best_Model_Prob",
        "edge_col": "Best_Edge",
        "odd_col": "Best_Odd",
        "target_col": "ResultTarget",
        "prediction_col": "Best_Result_Index",
        "positive_target": None,
    },
    "Vitoria Casa/Fora": {
        "prob_col": "Win_Model_Prob",
        "edge_col": "Win_Edge",
        "odd_col": "Win_Odd",
        "target_col": "ResultTarget",
        "prediction_col": "Win_Result_Index",
        "positive_target": None,
    },
}

EDGE_GRID = [0.00, 0.025, 0.05, 0.075, 0.10, 0.125]
OVER_PROB_GRID = [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70]
RESULT_PROB_GRID = [0.40, 0.45, 0.48, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625]
WIN_PROB_GRID = [0.45, 0.48, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65]
OVER_MAX_ODD_GRID = [1.60, 1.70, 1.80, 1.90, 2.00, 2.20, None]
RESULT_MAX_ODD_GRID = [1.50, 1.75, 2.00, 2.25, 2.50, 3.00, 4.00, None]
FILTER_OPTIMIZATION_SUMMARY_FILE = "filter_optimization_summary.csv"


def _market_grid(market: str) -> tuple[list[float], list[float], list[float | None]]:
    """Retorna a grade de busca adequada para o mercado."""
    if market in {"Over 2.5", "Under 2.5"}:
        return EDGE_GRID, OVER_PROB_GRID, OVER_MAX_ODD_GRID
    if market == "Resultado 1X2":
        return EDGE_GRID, RESULT_PROB_GRID, RESULT_MAX_ODD_GRID
    return EDGE_GRID, WIN_PROB_GRID, RESULT_MAX_ODD_GRID


def _hit_series(data: pd.DataFrame, market: str) -> pd.Series:
    """Calcula acerto de cada selecao simulada."""
    columns = MARKET_COLUMNS[market]
    target_col = columns["target_col"]
    prediction_col = columns["prediction_col"]

    if prediction_col is None:
        return data[target_col].eq(columns["positive_target"])
    return data[target_col].eq(data[prediction_col])


def _split_optimization_period(
    data: pd.DataFrame,
    train_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o backtest em ajuste e avaliacao preservando cronologia."""
    if not 0.0 < train_size < 1.0:
        raise ValueError("filter_optimization_train_size deve estar entre 0 e 1.")

    ordered = data.sort_values("MatchDatetime", kind="mergesort").reset_index(
        drop=True
    )
    split_index = int(len(ordered) * train_size)
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("Dados insuficientes para otimizar filtros.")
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def summarize_threshold_strategy(
    data: pd.DataFrame,
    market: str,
    edge_threshold: float,
    min_model_prob: float,
    max_odd: float | None,
    stake: float,
) -> dict[str, float]:
    """Simula uma regra de filtro e resume resultado financeiro."""
    columns = MARKET_COLUMNS[market]
    prob_col = columns["prob_col"]
    edge_col = columns["edge_col"]
    odd_col = columns["odd_col"]

    valid_data = data.dropna(subset=[prob_col, edge_col, odd_col]).copy()
    odd_filter = True if max_odd is None else valid_data[odd_col] <= max_odd
    bet_mask = (
        valid_data[edge_col].ge(edge_threshold)
        & valid_data[prob_col].ge(min_model_prob)
        & odd_filter
    )
    bets = valid_data[bet_mask].copy()

    if bets.empty:
        return {
            "Bets": 0.0,
            "TotalStaked": 0.0,
            "TotalProfit": 0.0,
            "ROI": 0.0,
            "HitRate": 0.0,
            "AvgEdge": 0.0,
            "AvgModelProb": 0.0,
            "AvgOdd": 0.0,
        }

    hits = _hit_series(bets, market)
    profits = np.where(hits, stake * (bets[odd_col].to_numpy() - 1.0), -stake)
    total_staked = float(len(bets) * stake)
    total_profit = float(profits.sum())

    return {
        "Bets": float(len(bets)),
        "TotalStaked": total_staked,
        "TotalProfit": total_profit,
        "ROI": total_profit / total_staked if total_staked else 0.0,
        "HitRate": float(hits.mean()),
        "AvgEdge": float(bets[edge_col].mean()),
        "AvgModelProb": float(bets[prob_col].mean()),
        "AvgOdd": float(bets[odd_col].mean()),
    }


def _prefixed_summary(prefix: str, summary: dict[str, float]) -> dict[str, float]:
    """Prefixa metricas de ajuste ou avaliacao."""
    return {f"{prefix}{key}": value for key, value in summary.items()}


def _period_bounds(data: pd.DataFrame) -> tuple[str, str]:
    """Retorna datas inicial/final ISO do periodo."""
    return (
        data["MatchDatetime"].min().date().isoformat(),
        data["MatchDatetime"].max().date().isoformat(),
    )


def _candidate_row(
    market: str,
    edge_threshold: float,
    min_model_prob: float,
    max_odd: float | None,
    tune_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    tune_summary: dict[str, float],
    eval_summary: dict[str, float],
    min_bets: int,
) -> dict[str, Any]:
    """Monta uma linha da grade de otimizacao."""
    tune_start, tune_end = _period_bounds(tune_data)
    eval_start, eval_end = _period_bounds(eval_data)

    return {
        "Market": market,
        "EdgeThreshold": edge_threshold,
        "MinModelProb": min_model_prob,
        "MaxOdd": np.nan if max_odd is None else max_odd,
        "HasMaxOdd": max_odd is not None,
        "TuneStart": tune_start,
        "TuneEnd": tune_end,
        "EvalStart": eval_start,
        "EvalEnd": eval_end,
        "Qualified": tune_summary["Bets"] >= min_bets,
        **_prefixed_summary("Tune", tune_summary),
        **_prefixed_summary("Eval", eval_summary),
    }


def optimize_market_filters(
    backtest: pd.DataFrame,
    market: str,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Busca filtros em janela inicial e avalia em janela posterior."""
    if backtest.empty:
        return pd.DataFrame(), pd.DataFrame()

    tune_data, eval_data = _split_optimization_period(
        backtest,
        config.filter_optimization_train_size,
    )
    edge_grid, prob_grid, max_odd_grid = _market_grid(market)
    rows = []

    for edge_threshold, min_model_prob, max_odd in product(
        edge_grid,
        prob_grid,
        max_odd_grid,
    ):
        tune_summary = summarize_threshold_strategy(
            tune_data,
            market,
            edge_threshold,
            min_model_prob,
            max_odd,
            config.stake,
        )
        eval_summary = summarize_threshold_strategy(
            eval_data,
            market,
            edge_threshold,
            min_model_prob,
            max_odd,
            config.stake,
        )
        rows.append(
            _candidate_row(
                market,
                edge_threshold,
                min_model_prob,
                max_odd,
                tune_data,
                eval_data,
                tune_summary,
                eval_summary,
                config.min_optimization_bets,
            )
        )

    grid = pd.DataFrame(rows)
    qualified = grid[grid["Qualified"]].copy()
    if qualified.empty:
        qualified = grid[grid["TuneBets"] > 0].copy()
    if qualified.empty:
        return pd.DataFrame(), grid

    best = qualified.sort_values(
        ["TuneROI", "TuneTotalProfit", "TuneBets"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(1)
    return best.reset_index(drop=True), grid


def print_filter_optimization_summary(summary: pd.DataFrame) -> None:
    """Imprime filtros otimizados por mercado."""
    if summary.empty:
        return

    print("\n========== OTIMIZACAO DE FILTROS ==========")
    for _, row in summary.iterrows():
        max_odd_text = (
            "sem limite" if pd.isna(row["MaxOdd"]) else f"{row['MaxOdd']:.2f}"
        )
        print(
            f"{row['Market']}: edge {row['EdgeThreshold']:.2%}, "
            f"prob min {row['MinModelProb']:.2%}, odd max {max_odd_text} | "
            f"ajuste ROI {row['TuneROI']:.2%} ({row['TuneBets']:.0f} apostas), "
            f"avaliacao ROI {row['EvalROI']:.2%} ({row['EvalBets']:.0f} apostas)"
        )


def _row_max_odd(row: pd.Series) -> float | None:
    """Converte a odd maxima salva no CSV para None quando nao ha limite."""
    if "HasMaxOdd" in row.index and not bool(row["HasMaxOdd"]):
        return None
    if pd.isna(row["MaxOdd"]):
        return None
    return float(row["MaxOdd"])


def load_positive_filter_rules(
    output_dir: Path,
    min_eval_roi: float = 0.0,
    min_eval_bets: int = 10,
) -> dict[str, dict[str, float | str | None]]:
    """Carrega regras otimizadas que passaram na avaliacao futura.

    Apenas regras com ROI de avaliacao acima do minimo e volume suficiente
    sao liberadas para uso em palpites futuros.
    """
    path = output_dir / FILTER_OPTIMIZATION_SUMMARY_FILE
    if not path.exists():
        return {}

    try:
        summary = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}

    required_cols = {
        "Market",
        "EdgeThreshold",
        "MinModelProb",
        "MaxOdd",
        "EvalROI",
        "EvalBets",
    }
    if not required_cols.issubset(summary.columns):
        return {}

    usable = summary[
        summary["EvalROI"].gt(min_eval_roi)
        & summary["EvalBets"].ge(min_eval_bets)
    ].copy()
    if usable.empty:
        return {}

    usable = usable.sort_values(
        ["Market", "EvalROI", "EvalTotalProfit", "EvalBets"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    rules: dict[str, dict[str, float | str | None]] = {}
    for market, market_rows in usable.groupby("Market", sort=False):
        row = market_rows.iloc[0]
        rules[str(market)] = {
            "source": "Otimizada",
            "edge_threshold": float(row["EdgeThreshold"]),
            "min_model_prob": float(row["MinModelProb"]),
            "max_odd": _row_max_odd(row),
            "eval_roi": float(row["EvalROI"]),
            "eval_bets": float(row["EvalBets"]),
        }

    return rules
