"""Backtests financeiros, relatorios e validacao walk-forward."""

from __future__ import annotations

from typing import Sequence

import numpy as np
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
    PipelineConfig,
    RESULT_LABELS,
    RESULT_NAME_MAP,
    RESULT_TARGET_COL,
    TARGET_COL,
)
from data_pipeline import add_no_vig_market_probabilities
from modeling import (
    blend_probabilities,
    build_time_decay_sample_weights,
    evaluate_match_result_model,
    evaluate_model,
    train_calibrated_xgboost_model,
    tune_probability_blend,
)


def run_backtest(
    test_data: pd.DataFrame,
    probabilities: np.ndarray,
    stake: float = 10.0,
    edge: float = 0.05,
    min_model_prob: float = 0.55,
    max_over_odd: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simula apostas fixas no Over 2.5 quando ha edge minimo.

    Regra:
        aposta se prob_modelo > prob_no_vig_casa + edge
    """
    backtest = test_data.copy()
    if NO_VIG_OVER_COL not in backtest.columns:
        backtest = add_no_vig_market_probabilities(backtest)

    backtest["Model_Prob_Over25"] = probabilities
    backtest["Edge"] = (
        backtest["Model_Prob_Over25"] - backtest[NO_VIG_OVER_COL]
    )
    backtest["Pass_Edge_Filter"] = backtest["Edge"] >= edge
    backtest["Pass_Prob_Filter"] = backtest["Model_Prob_Over25"] >= min_model_prob
    if max_over_odd is None:
        backtest["Pass_Odd_Filter"] = True
    else:
        backtest["Pass_Odd_Filter"] = backtest[ODD_OVER_COL] <= max_over_odd

    backtest["Bet_Over25"] = (
        backtest["Pass_Edge_Filter"]
        & backtest["Pass_Prob_Filter"]
        & backtest["Pass_Odd_Filter"]
    )
    backtest["Stake"] = np.where(backtest["Bet_Over25"], stake, 0.0)
    backtest["Profit"] = np.where(
        ~backtest["Bet_Over25"],
        0.0,
        np.where(
            backtest[TARGET_COL].eq(1),
            stake * (backtest[ODD_OVER_COL] - 1.0),
            -stake,
        ),
    )

    bets = backtest[backtest["Bet_Over25"]].copy()
    total_staked = float(bets["Stake"].sum())
    total_profit = float(bets["Profit"].sum())
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    hit_rate = float(bets[TARGET_COL].mean()) if len(bets) else 0.0
    avg_edge = float(bets["Edge"].mean()) if len(bets) else 0.0
    avg_model_prob = float(bets["Model_Prob_Over25"].mean()) if len(bets) else 0.0
    avg_odd = float(bets[ODD_OVER_COL].mean()) if len(bets) else 0.0

    summary = {
        "bets": float(len(bets)),
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_edge": avg_edge,
        "avg_model_prob": avg_model_prob,
        "avg_odd": avg_odd,
    }

    return backtest, summary


def run_under25_backtest(
    test_data: pd.DataFrame,
    over_probabilities: np.ndarray,
    stake: float = 10.0,
    edge: float = 0.05,
    min_model_prob: float = 0.55,
    max_under_odd: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simula apostas fixas no Under 2.5 usando 1 - probabilidade de Over."""
    backtest = test_data.copy()
    if NO_VIG_UNDER_COL not in backtest.columns:
        backtest = add_no_vig_market_probabilities(backtest)

    backtest["Model_Prob_Over25"] = over_probabilities
    backtest["Model_Prob_Under25"] = 1.0 - over_probabilities
    backtest["Under25"] = backtest[TARGET_COL].eq(0).astype(int)
    backtest["Under_Edge"] = (
        backtest["Model_Prob_Under25"] - backtest[NO_VIG_UNDER_COL]
    )
    backtest["Pass_Edge_Filter"] = backtest["Under_Edge"] >= edge
    backtest["Pass_Prob_Filter"] = (
        backtest["Model_Prob_Under25"] >= min_model_prob
    )
    if max_under_odd is None:
        backtest["Pass_Odd_Filter"] = True
    else:
        backtest["Pass_Odd_Filter"] = backtest[ODD_UNDER_COL] <= max_under_odd

    backtest["Bet_Under25"] = (
        backtest["Pass_Edge_Filter"]
        & backtest["Pass_Prob_Filter"]
        & backtest["Pass_Odd_Filter"]
    )
    backtest["Stake"] = np.where(backtest["Bet_Under25"], stake, 0.0)
    backtest["Profit"] = np.where(
        ~backtest["Bet_Under25"],
        0.0,
        np.where(
            backtest[TARGET_COL].eq(0),
            stake * (backtest[ODD_UNDER_COL] - 1.0),
            -stake,
        ),
    )

    bets = backtest[backtest["Bet_Under25"]].copy()
    total_staked = float(bets["Stake"].sum())
    total_profit = float(bets["Profit"].sum())
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    hit_rate = float(bets["Under25"].mean()) if len(bets) else 0.0
    avg_edge = float(bets["Under_Edge"].mean()) if len(bets) else 0.0
    avg_model_prob = (
        float(bets["Model_Prob_Under25"].mean()) if len(bets) else 0.0
    )
    avg_odd = float(bets[ODD_UNDER_COL].mean()) if len(bets) else 0.0

    summary = {
        "bets": float(len(bets)),
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_edge": avg_edge,
        "avg_model_prob": avg_model_prob,
        "avg_odd": avg_odd,
    }

    return backtest, summary


def run_match_result_backtest(
    test_data: pd.DataFrame,
    probabilities: np.ndarray,
    stake: float = 10.0,
    edge: float = 0.05,
    min_model_prob: float = 0.48,
    max_result_odd: float | None = 2.50,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simula apostas +EV no mercado Resultado Final 1X2."""
    backtest = test_data.copy()
    if NO_VIG_HOME_COL not in backtest.columns:
        backtest = add_no_vig_market_probabilities(backtest)

    odd_cols = [ODD_HOME_COL, ODD_DRAW_COL, ODD_AWAY_COL]
    no_vig_cols = [NO_VIG_HOME_COL, NO_VIG_DRAW_COL, NO_VIG_AWAY_COL]
    odds = backtest.loc[:, odd_cols].to_numpy(dtype=float)
    no_vig_probs = backtest.loc[:, no_vig_cols].to_numpy(dtype=float)
    edges = probabilities - no_vig_probs
    best_indices = np.argmax(edges, axis=1)
    row_indices = np.arange(len(backtest))

    for index, label in enumerate(RESULT_LABELS):
        backtest[f"Model_Prob_{label}"] = probabilities[:, index]
        backtest[f"NoVig_Prob_{label}"] = no_vig_probs[:, index]
        backtest[f"Edge_{label}"] = edges[:, index]

    backtest["Best_Result_Index"] = best_indices
    backtest["Best_Result"] = [RESULT_LABELS[index] for index in best_indices]
    backtest["Best_Result_Name"] = backtest["Best_Result"].map(RESULT_NAME_MAP)
    backtest["Best_Model_Prob"] = probabilities[row_indices, best_indices]
    backtest["Best_NoVig_Prob"] = no_vig_probs[row_indices, best_indices]
    backtest["Best_Edge"] = edges[row_indices, best_indices]
    backtest["Best_Odd"] = odds[row_indices, best_indices]
    backtest["Actual_Result"] = backtest["FTR"]

    backtest["Pass_Edge_Filter"] = backtest["Best_Edge"] >= edge
    backtest["Pass_Prob_Filter"] = backtest["Best_Model_Prob"] >= min_model_prob
    if max_result_odd is None:
        backtest["Pass_Odd_Filter"] = True
    else:
        backtest["Pass_Odd_Filter"] = backtest["Best_Odd"] <= max_result_odd

    backtest["Bet_Result"] = (
        backtest["Pass_Edge_Filter"]
        & backtest["Pass_Prob_Filter"]
        & backtest["Pass_Odd_Filter"]
    )
    backtest["Stake"] = np.where(backtest["Bet_Result"], stake, 0.0)
    backtest["Profit"] = np.where(
        ~backtest["Bet_Result"],
        0.0,
        np.where(
            backtest[RESULT_TARGET_COL].eq(backtest["Best_Result_Index"]),
            stake * (backtest["Best_Odd"] - 1.0),
            -stake,
        ),
    )

    bets = backtest[backtest["Bet_Result"]].copy()
    total_staked = float(bets["Stake"].sum())
    total_profit = float(bets["Profit"].sum())
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    hit_rate = (
        float(bets[RESULT_TARGET_COL].eq(bets["Best_Result_Index"]).mean())
        if len(bets)
        else 0.0
    )
    avg_edge = float(bets["Best_Edge"].mean()) if len(bets) else 0.0
    avg_model_prob = float(bets["Best_Model_Prob"].mean()) if len(bets) else 0.0
    avg_odd = float(bets["Best_Odd"].mean()) if len(bets) else 0.0

    summary = {
        "bets": float(len(bets)),
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_edge": avg_edge,
        "avg_model_prob": avg_model_prob,
        "avg_odd": avg_odd,
    }

    return backtest, summary


def run_win_backtest(
    test_data: pd.DataFrame,
    probabilities: np.ndarray,
    stake: float = 10.0,
    edge: float = 0.05,
    min_model_prob: float = 0.50,
    max_win_odd: float | None = 2.50,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simula apostas +EV apenas em vitoria do mandante ou visitante."""
    backtest = test_data.copy()
    if NO_VIG_HOME_COL not in backtest.columns:
        backtest = add_no_vig_market_probabilities(backtest)

    candidate_indices = np.array([0, 2])
    odd_cols = [ODD_HOME_COL, ODD_AWAY_COL]
    no_vig_cols = [NO_VIG_HOME_COL, NO_VIG_AWAY_COL]
    candidate_probs = probabilities[:, candidate_indices]
    candidate_no_vig = backtest.loc[:, no_vig_cols].to_numpy(dtype=float)
    candidate_odds = backtest.loc[:, odd_cols].to_numpy(dtype=float)
    candidate_edges = candidate_probs - candidate_no_vig
    best_candidate_positions = np.argmax(candidate_edges, axis=1)
    best_indices = candidate_indices[best_candidate_positions]
    row_indices = np.arange(len(backtest))

    backtest["Model_Prob_H"] = probabilities[:, 0]
    backtest["NoVig_Prob_H"] = backtest[NO_VIG_HOME_COL]
    backtest["Edge_H"] = probabilities[:, 0] - backtest[NO_VIG_HOME_COL]
    backtest["Model_Prob_A"] = probabilities[:, 2]
    backtest["NoVig_Prob_A"] = backtest[NO_VIG_AWAY_COL]
    backtest["Edge_A"] = probabilities[:, 2] - backtest[NO_VIG_AWAY_COL]
    backtest["Win_Result_Index"] = best_indices
    backtest["Win_Result"] = [RESULT_LABELS[index] for index in best_indices]
    backtest["Win_Result_Name"] = backtest["Win_Result"].map(RESULT_NAME_MAP)
    backtest["Win_Model_Prob"] = candidate_probs[
        row_indices,
        best_candidate_positions,
    ]
    backtest["Win_NoVig_Prob"] = candidate_no_vig[
        row_indices,
        best_candidate_positions,
    ]
    backtest["Win_Edge"] = candidate_edges[row_indices, best_candidate_positions]
    backtest["Win_Odd"] = candidate_odds[row_indices, best_candidate_positions]
    backtest["Actual_Result"] = backtest["FTR"]

    backtest["Pass_Edge_Filter"] = backtest["Win_Edge"] >= edge
    backtest["Pass_Prob_Filter"] = backtest["Win_Model_Prob"] >= min_model_prob
    if max_win_odd is None:
        backtest["Pass_Odd_Filter"] = True
    else:
        backtest["Pass_Odd_Filter"] = backtest["Win_Odd"] <= max_win_odd

    backtest["Bet_Win"] = (
        backtest["Pass_Edge_Filter"]
        & backtest["Pass_Prob_Filter"]
        & backtest["Pass_Odd_Filter"]
    )
    backtest["Stake"] = np.where(backtest["Bet_Win"], stake, 0.0)
    backtest["Profit"] = np.where(
        ~backtest["Bet_Win"],
        0.0,
        np.where(
            backtest[RESULT_TARGET_COL].eq(backtest["Win_Result_Index"]),
            stake * (backtest["Win_Odd"] - 1.0),
            -stake,
        ),
    )

    bets = backtest[backtest["Bet_Win"]].copy()
    total_staked = float(bets["Stake"].sum())
    total_profit = float(bets["Profit"].sum())
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    hit_rate = (
        float(bets[RESULT_TARGET_COL].eq(bets["Win_Result_Index"]).mean())
        if len(bets)
        else 0.0
    )
    avg_edge = float(bets["Win_Edge"].mean()) if len(bets) else 0.0
    avg_model_prob = float(bets["Win_Model_Prob"].mean()) if len(bets) else 0.0
    avg_odd = float(bets["Win_Odd"].mean()) if len(bets) else 0.0

    summary = {
        "bets": float(len(bets)),
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_edge": avg_edge,
        "avg_model_prob": avg_model_prob,
        "avg_odd": avg_odd,
    }

    return backtest, summary


def print_results(
    metrics: dict[str, float],
    backtest_summary: dict[str, float],
    stake: float,
    edge: float,
    min_model_prob: float,
    max_over_odd: float | None,
) -> None:
    """Imprime as metricas finais de modelo e simulacao financeira."""
    print("\n========== RESULTADOS DO MODELO ==========")
    print(f"Acuracia:                 {metrics['accuracy']:.2%}")
    print(f"Precisao ao prever Over:  {metrics['precision_over']:.2%}")
    print(f"LogLoss:                  {metrics['logloss']:.4f}")
    if "brier_score" in metrics:
        print(f"Brier Score:              {metrics['brier_score']:.4f}")
    if "calibration_ece" in metrics:
        print(f"Erro calibracao (ECE):    {metrics['calibration_ece']:.2%}")

    print("\n========== BACKTEST OVER 2.5 +EV ==========")
    print("Prob. referencia:         No-vig Over 2.5")
    print(f"Stake fixa:               R$ {stake:.2f}")
    print(f"Edge minimo:              {edge:.2%}")
    print(f"Prob. minima modelo:      {min_model_prob:.2%}")
    if max_over_odd is not None:
        print(f"Odd maxima Over:          {max_over_odd:.2f}")
    print(f"Apostas simuladas:        {backtest_summary['bets']:.0f}")
    print(f"Total apostado:           R$ {backtest_summary['total_staked']:.2f}")
    print(f"Lucro/Prejuizo:           R$ {backtest_summary['total_profit']:.2f}")
    print(f"ROI:                      {backtest_summary['roi']:.2%}")
    print(f"Taxa de acerto apostas:   {backtest_summary['hit_rate']:.2%}")
    print(f"Edge medio das apostas:   {backtest_summary['avg_edge']:.2%}")
    print(f"Prob. media modelo:       {backtest_summary['avg_model_prob']:.2%}")
    print(f"Odd media das apostas:    {backtest_summary['avg_odd']:.2f}")


def print_under25_results(
    metrics: dict[str, float],
    backtest_summary: dict[str, float],
    stake: float,
    edge: float,
    min_model_prob: float,
    max_under_odd: float | None,
) -> None:
    """Imprime metricas finais do mercado Under 2.5."""
    print("\n========== RESULTADOS DO MODELO ==========")
    print(f"Acuracia:                 {metrics['accuracy']:.2%}")
    print(f"Precisao ao prever Over:  {metrics['precision_over']:.2%}")
    print(f"LogLoss:                  {metrics['logloss']:.4f}")
    if "brier_score" in metrics:
        print(f"Brier Score:              {metrics['brier_score']:.4f}")
    if "calibration_ece" in metrics:
        print(f"Erro calibracao (ECE):    {metrics['calibration_ece']:.2%}")

    print("\n========== BACKTEST UNDER 2.5 +EV ==========")
    print("Prob. referencia:         No-vig Under 2.5")
    print(f"Stake fixa:               R$ {stake:.2f}")
    print(f"Edge minimo:              {edge:.2%}")
    print(f"Prob. minima modelo:      {min_model_prob:.2%}")
    if max_under_odd is not None:
        print(f"Odd maxima Under:         {max_under_odd:.2f}")
    print(f"Apostas simuladas:        {backtest_summary['bets']:.0f}")
    print(f"Total apostado:           R$ {backtest_summary['total_staked']:.2f}")
    print(f"Lucro/Prejuizo:           R$ {backtest_summary['total_profit']:.2f}")
    print(f"ROI:                      {backtest_summary['roi']:.2%}")
    print(f"Taxa de acerto apostas:   {backtest_summary['hit_rate']:.2%}")
    print(f"Edge medio das apostas:   {backtest_summary['avg_edge']:.2%}")
    print(f"Prob. media modelo:       {backtest_summary['avg_model_prob']:.2%}")
    print(f"Odd media das apostas:    {backtest_summary['avg_odd']:.2f}")


def print_match_result_results(
    metrics: dict[str, float],
    backtest_summary: dict[str, float],
    stake: float,
    edge: float,
    min_model_prob: float,
    max_result_odd: float | None,
) -> None:
    """Imprime metricas finais do mercado Resultado Final 1X2."""
    print("\n========== RESULTADOS DO MODELO 1X2 ==========")
    print(f"Acuracia:                 {metrics['accuracy']:.2%}")
    print(f"Precisao Casa:            {metrics['precision_h']:.2%}")
    print(f"Precisao Empate:          {metrics['precision_d']:.2%}")
    print(f"Precisao Fora:            {metrics['precision_a']:.2%}")
    print(f"LogLoss:                  {metrics['logloss']:.4f}")
    if "brier_score" in metrics:
        print(f"Brier Score:              {metrics['brier_score']:.4f}")
    if "calibration_ece" in metrics:
        print(f"Erro calibracao (ECE):    {metrics['calibration_ece']:.2%}")

    print("\n========== BACKTEST RESULTADO FINAL 1X2 +EV ==========")
    print("Prob. referencia:         No-vig 1X2")
    print(f"Stake fixa:               R$ {stake:.2f}")
    print(f"Edge minimo:              {edge:.2%}")
    print(f"Prob. minima modelo:      {min_model_prob:.2%}")
    if max_result_odd is not None:
        print(f"Odd maxima 1X2:           {max_result_odd:.2f}")
    print(f"Apostas simuladas:        {backtest_summary['bets']:.0f}")
    print(f"Total apostado:           R$ {backtest_summary['total_staked']:.2f}")
    print(f"Lucro/Prejuizo:           R$ {backtest_summary['total_profit']:.2f}")
    print(f"ROI:                      {backtest_summary['roi']:.2%}")
    print(f"Taxa de acerto apostas:   {backtest_summary['hit_rate']:.2%}")
    print(f"Edge medio das apostas:   {backtest_summary['avg_edge']:.2%}")
    print(f"Prob. media modelo:       {backtest_summary['avg_model_prob']:.2%}")
    print(f"Odd media das apostas:    {backtest_summary['avg_odd']:.2f}")


def print_win_results(
    metrics: dict[str, float],
    backtest_summary: dict[str, float],
    stake: float,
    edge: float,
    min_model_prob: float,
    max_win_odd: float | None,
) -> None:
    """Imprime metricas finais do mercado de Vitoria Casa/Fora."""
    print("\n========== RESULTADOS DO MODELO VITORIA ==========")
    print(f"Acuracia 1X2 base:        {metrics['accuracy']:.2%}")
    print(f"Precisao Casa:            {metrics['precision_h']:.2%}")
    print(f"Precisao Fora:            {metrics['precision_a']:.2%}")
    print(f"LogLoss 1X2 base:         {metrics['logloss']:.4f}")
    if "brier_score" in metrics:
        print(f"Brier Score 1X2:          {metrics['brier_score']:.4f}")
    if "calibration_ece" in metrics:
        print(f"Erro calibracao (ECE):    {metrics['calibration_ece']:.2%}")

    print("\n========== BACKTEST VITORIA CASA/FORA +EV ==========")
    print("Prob. referencia:         No-vig 1X2")
    print(f"Stake fixa:               R$ {stake:.2f}")
    print(f"Edge minimo:              {edge:.2%}")
    print(f"Prob. minima modelo:      {min_model_prob:.2%}")
    if max_win_odd is not None:
        print(f"Odd maxima vitoria:       {max_win_odd:.2f}")
    print(f"Apostas simuladas:        {backtest_summary['bets']:.0f}")
    print(f"Total apostado:           R$ {backtest_summary['total_staked']:.2f}")
    print(f"Lucro/Prejuizo:           R$ {backtest_summary['total_profit']:.2f}")
    print(f"ROI:                      {backtest_summary['roi']:.2%}")
    print(f"Taxa de acerto apostas:   {backtest_summary['hit_rate']:.2%}")
    print(f"Edge medio das apostas:   {backtest_summary['avg_edge']:.2%}")
    print(f"Prob. media modelo:       {backtest_summary['avg_model_prob']:.2%}")
    print(f"Odd media das apostas:    {backtest_summary['avg_odd']:.2f}")


def _has_required_classes(
    target: pd.Series,
    required_classes: Sequence[int],
) -> bool:
    """Verifica se uma janela contem as classes necessarias."""
    available = set(pd.Series(target).dropna().astype(int).unique().tolist())
    return set(required_classes).issubset(available)


def _build_walk_forward_date_folds(
    data: pd.DataFrame,
    config: PipelineConfig,
    initial_train_fraction: float,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Cria folds expansivos por blocos de datas, sem misturar o mesmo dia."""
    if config.walk_forward_splits <= 0:
        return []

    if not 0.2 <= initial_train_fraction < 1.0:
        raise ValueError("initial_train_fraction deve estar entre 0.2 e 1.")

    ordered = data.sort_values("MatchDatetime", kind="mergesort").reset_index(
        drop=True
    )
    ordered["_WalkForwardDate"] = pd.to_datetime(
        ordered["MatchDatetime"],
        errors="coerce",
    ).dt.normalize()
    ordered = ordered.dropna(subset=["_WalkForwardDate"]).copy()
    unique_dates = np.array(sorted(ordered["_WalkForwardDate"].unique()))
    initial_date_count = int(len(unique_dates) * initial_train_fraction)

    if initial_date_count <= 0 or initial_date_count >= len(unique_dates):
        raise ValueError("Datas insuficientes para walk-forward validation.")

    remaining_dates = unique_dates[initial_date_count:]
    date_blocks = [
        block for block in np.array_split(remaining_dates, config.walk_forward_splits)
        if len(block)
    ]
    folds: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []

    for fold, block in enumerate(date_blocks, start=1):
        test_start = pd.Timestamp(block[0])
        test_end = pd.Timestamp(block[-1])
        train_mask = ordered["_WalkForwardDate"] < test_start
        test_mask = (
            ordered["_WalkForwardDate"].ge(test_start)
            & ordered["_WalkForwardDate"].le(test_end)
        )
        train_data = ordered.loc[train_mask].drop(columns="_WalkForwardDate").copy()
        test_data = ordered.loc[test_mask].drop(columns="_WalkForwardDate").copy()

        if len(test_data) < config.walk_forward_min_test_rows:
            print(
                "[walk-forward] Fold ignorado por baixo volume: "
                f"{fold} ({len(test_data):,} jogos)"
            )
            continue
        if train_data.empty or test_data.empty:
            continue

        folds.append((fold, train_data, test_data))

    if not folds:
        raise ValueError("Nenhum fold walk-forward valido foi criado.")

    return folds


def _split_walk_forward_model_train(
    train_data: pd.DataFrame,
    target_col: str,
    config: PipelineConfig,
    required_classes: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino do fold e uma validacao interna para blend."""
    split_index = int(len(train_data) * (1.0 - config.calibration_size))
    if split_index <= 0 or split_index >= len(train_data):
        return train_data.copy(), pd.DataFrame()

    model_train = train_data.iloc[:split_index].copy()
    blend_validation = train_data.iloc[split_index:].copy()
    if not _has_required_classes(model_train[target_col], required_classes):
        return train_data.copy(), pd.DataFrame()
    if not _has_required_classes(blend_validation[target_col], required_classes):
        return train_data.copy(), pd.DataFrame()
    return model_train, blend_validation


def _walk_forward_sample_weights(
    train_data: pd.DataFrame,
    config: PipelineConfig,
) -> np.ndarray | None:
    """Calcula pesos temporais para o treino de um fold."""
    if not config.use_time_decay_weights:
        return None
    return build_time_decay_sample_weights(
        train_data["MatchDatetime"],
        half_life_days=config.time_decay_half_life_days,
        min_weight=config.min_time_decay_weight,
    )


def _market_probability_values(
    data: pd.DataFrame,
    columns: str | Sequence[str],
) -> np.ndarray | None:
    """Extrai probabilidades no-vig de mercado quando estao disponiveis."""
    if isinstance(columns, str):
        if columns not in data.columns:
            return None
        return data[columns].to_numpy(dtype=float)

    column_list = list(columns)
    if any(column not in data.columns for column in column_list):
        return None
    return data[column_list].to_numpy(dtype=float)


def _blend_fold_probabilities(
    y_validation: pd.Series,
    validation_model_probabilities: np.ndarray,
    test_model_probabilities: np.ndarray,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    market_probability_columns: str | Sequence[str],
    labels: list[int],
    market: str,
) -> tuple[np.ndarray, float, int]:
    """Ajusta blend modelo x mercado dentro do fold, sem olhar o teste."""
    validation_market_probabilities = _market_probability_values(
        validation_data,
        market_probability_columns,
    )
    test_market_probabilities = _market_probability_values(
        test_data,
        market_probability_columns,
    )
    if validation_market_probabilities is None or test_market_probabilities is None:
        return test_model_probabilities, 1.0, 0

    alpha_model, _ = tune_probability_blend(
        y_validation,
        validation_model_probabilities,
        validation_market_probabilities,
        labels=labels,
        market=f"{market} | fold",
    )
    probabilities = blend_probabilities(
        test_model_probabilities,
        test_market_probabilities,
        alpha_model,
    )
    return probabilities, alpha_model, len(validation_data)


def _base_walk_forward_row(
    market: str,
    fold: int,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    alpha_model: float,
    blend_validation_rows: int,
) -> dict[str, float | int | str]:
    """Monta metadados comuns de um fold walk-forward."""
    train_start = train_data["MatchDatetime"].min()
    train_end = train_data["MatchDatetime"].max()
    test_start = test_data["MatchDatetime"].min()
    test_end = test_data["MatchDatetime"].max()
    return {
        "Market": market,
        "fold": fold,
        "train_start": train_start.date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_end": test_end.date().isoformat(),
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "alpha_model": alpha_model,
        "alpha_market": 1.0 - alpha_model,
        "blend_validation_rows": blend_validation_rows,
    }


def _print_walk_forward_summary(summary: pd.DataFrame, title: str) -> None:
    """Imprime resumo agregado dos folds."""
    total_staked = float(summary["total_staked"].sum())
    total_profit = float(summary["total_profit"].sum())
    total_bets = float(summary["bets"].sum())
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    weighted_hit_rate = (
        float((summary["hit_rate"] * summary["bets"]).sum() / total_bets)
        if total_bets > 0
        else 0.0
    )
    positive_folds = float(summary["roi"].gt(0).mean()) if len(summary) else 0.0

    print(f"\n========== RESUMO WALK-FORWARD {title} ==========")
    print(f"Folds:                    {len(summary)}")
    print(f"Folds ROI positivo:       {positive_folds:.2%}")
    print(f"Acuracia media:           {summary['accuracy'].mean():.2%}")
    print(f"LogLoss medio:            {summary['logloss'].mean():.4f}")
    print(f"Brier medio:              {summary['brier_score'].mean():.4f}")
    print(f"ECE medio:                {summary['calibration_ece'].mean():.2%}")
    print(f"Apostas simuladas:        {summary['bets'].sum():.0f}")
    print(f"Taxa de acerto apostas:   {weighted_hit_rate:.2%}")
    print(f"Total apostado:           R$ {total_staked:.2f}")
    print(f"Lucro/Prejuizo:           R$ {total_profit:.2f}")
    print(f"ROI agregado:             {roi:.2%}")


def run_walk_forward_validation(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    config: PipelineConfig,
    initial_train_fraction: float | None = None,
) -> pd.DataFrame:
    """Executa validacao walk-forward por blocos de datas para Over 2.5."""
    if config.walk_forward_splits <= 0:
        return pd.DataFrame()

    train_fraction = (
        config.walk_forward_initial_train_fraction
        if initial_train_fraction is None
        else initial_train_fraction
    )
    folds = _build_walk_forward_date_folds(data, config, train_fraction)
    fold_rows: list[dict[str, float | int | str]] = []
    print("\n========== WALK-FORWARD VALIDATION OVER 2.5 ==========")

    for fold, train_data, test_data in folds:
        if not _has_required_classes(train_data[TARGET_COL], [0, 1]):
            print(f"[walk-forward Over 2.5] Fold {fold} ignorado: treino sem 2 classes.")
            continue

        model_train, blend_validation = _split_walk_forward_model_train(
            train_data,
            TARGET_COL,
            config,
            [0, 1],
        )
        print(
            f"[walk-forward Over 2.5] Fold {fold}/{len(folds)}: "
            f"treino ate {train_data['MatchDatetime'].max().date()}, "
            f"teste {test_data['MatchDatetime'].min().date()} a "
            f"{test_data['MatchDatetime'].max().date()} "
            f"({len(test_data):,} jogos)"
        )
        model = train_calibrated_xgboost_model(
            model_train.loc[:, feature_cols],
            model_train[TARGET_COL],
            config.calibration_size,
            config.calibration_method,
            config.xgb_tuning_trials,
            config.xgb_tuning_validation_size,
            sample_weight=_walk_forward_sample_weights(model_train, config),
        )
        probabilities = model.predict_proba(test_data.loc[:, feature_cols])[:, 1]
        alpha_model = 1.0
        blend_validation_rows = 0
        if not blend_validation.empty:
            validation_probabilities = model.predict_proba(
                blend_validation.loc[:, feature_cols]
            )[:, 1]
            probabilities, alpha_model, blend_validation_rows = (
                _blend_fold_probabilities(
                    blend_validation[TARGET_COL],
                    validation_probabilities,
                    probabilities,
                    blend_validation,
                    test_data,
                    NO_VIG_OVER_COL,
                    [0, 1],
                    "Over 2.5",
                )
            )

        metrics = evaluate_model(test_data[TARGET_COL], probabilities)
        _, backtest_summary = run_backtest(
            test_data,
            probabilities,
            stake=config.stake,
            edge=config.edge,
            min_model_prob=config.min_model_prob,
            max_over_odd=config.max_over_odd,
        )
        row = _base_walk_forward_row(
            "Over 2.5",
            fold,
            train_data,
            test_data,
            alpha_model,
            blend_validation_rows,
        )
        row.update(
            {
                "accuracy": metrics["accuracy"],
                "precision_over": metrics["precision_over"],
                "logloss": metrics["logloss"],
                "brier_score": metrics["brier_score"],
                "calibration_ece": metrics["calibration_ece"],
                **backtest_summary,
            }
        )
        fold_rows.append(row)

    summary = pd.DataFrame(fold_rows)
    if not summary.empty:
        _print_walk_forward_summary(summary, "OVER 2.5")
    return summary


def run_under25_walk_forward_validation(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    config: PipelineConfig,
    initial_train_fraction: float | None = None,
) -> pd.DataFrame:
    """Executa validacao walk-forward por blocos de datas para Under 2.5."""
    if config.walk_forward_splits <= 0:
        return pd.DataFrame()

    train_fraction = (
        config.walk_forward_initial_train_fraction
        if initial_train_fraction is None
        else initial_train_fraction
    )
    folds = _build_walk_forward_date_folds(data, config, train_fraction)
    fold_rows: list[dict[str, float | int | str]] = []
    print("\n========== WALK-FORWARD VALIDATION UNDER 2.5 ==========")

    for fold, train_data, test_data in folds:
        if not _has_required_classes(train_data[TARGET_COL], [0, 1]):
            print(f"[walk-forward Under 2.5] Fold {fold} ignorado: treino sem 2 classes.")
            continue

        model_train, blend_validation = _split_walk_forward_model_train(
            train_data,
            TARGET_COL,
            config,
            [0, 1],
        )
        print(
            f"[walk-forward Under 2.5] Fold {fold}/{len(folds)}: "
            f"treino ate {train_data['MatchDatetime'].max().date()}, "
            f"teste {test_data['MatchDatetime'].min().date()} a "
            f"{test_data['MatchDatetime'].max().date()} "
            f"({len(test_data):,} jogos)"
        )
        model = train_calibrated_xgboost_model(
            model_train.loc[:, feature_cols],
            model_train[TARGET_COL],
            config.calibration_size,
            config.calibration_method,
            config.xgb_tuning_trials,
            config.xgb_tuning_validation_size,
            sample_weight=_walk_forward_sample_weights(model_train, config),
        )
        under_probabilities = 1.0 - model.predict_proba(
            test_data.loc[:, feature_cols]
        )[:, 1]
        alpha_model = 1.0
        blend_validation_rows = 0
        if not blend_validation.empty:
            validation_under_probabilities = 1.0 - model.predict_proba(
                blend_validation.loc[:, feature_cols]
            )[:, 1]
            under_probabilities, alpha_model, blend_validation_rows = (
                _blend_fold_probabilities(
                    blend_validation[TARGET_COL].eq(0).astype(int),
                    validation_under_probabilities,
                    under_probabilities,
                    blend_validation,
                    test_data,
                    NO_VIG_UNDER_COL,
                    [0, 1],
                    "Under 2.5",
                )
            )

        y_under = test_data[TARGET_COL].eq(0).astype(int)
        metrics = evaluate_model(y_under, under_probabilities)
        over_probabilities = 1.0 - under_probabilities
        _, backtest_summary = run_under25_backtest(
            test_data,
            over_probabilities,
            stake=config.stake,
            edge=config.edge,
            min_model_prob=config.min_under_prob,
            max_under_odd=config.max_under_odd,
        )
        row = _base_walk_forward_row(
            "Under 2.5",
            fold,
            train_data,
            test_data,
            alpha_model,
            blend_validation_rows,
        )
        row.update(
            {
                "accuracy": metrics["accuracy"],
                "precision_under": metrics["precision_over"],
                "logloss": metrics["logloss"],
                "brier_score": metrics["brier_score"],
                "calibration_ece": metrics["calibration_ece"],
                **backtest_summary,
            }
        )
        fold_rows.append(row)

    summary = pd.DataFrame(fold_rows)
    if not summary.empty:
        _print_walk_forward_summary(summary, "UNDER 2.5")
    return summary


def run_match_result_walk_forward_validation(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    config: PipelineConfig,
    initial_train_fraction: float | None = None,
) -> pd.DataFrame:
    """Executa validacao walk-forward por blocos de datas para Resultado 1X2."""
    if config.walk_forward_splits <= 0:
        return pd.DataFrame()

    train_fraction = (
        config.walk_forward_initial_train_fraction
        if initial_train_fraction is None
        else initial_train_fraction
    )
    folds = _build_walk_forward_date_folds(data, config, train_fraction)
    fold_rows: list[dict[str, float | int | str]] = []
    print("\n========== WALK-FORWARD VALIDATION 1X2 ==========")

    for fold, train_data, test_data in folds:
        if not _has_required_classes(train_data[RESULT_TARGET_COL], [0, 1, 2]):
            print(f"[walk-forward 1X2] Fold {fold} ignorado: treino sem 3 classes.")
            continue

        model_train, blend_validation = _split_walk_forward_model_train(
            train_data,
            RESULT_TARGET_COL,
            config,
            [0, 1, 2],
        )
        print(
            f"[walk-forward 1X2] Fold {fold}/{len(folds)}: "
            f"treino ate {train_data['MatchDatetime'].max().date()}, "
            f"teste {test_data['MatchDatetime'].min().date()} a "
            f"{test_data['MatchDatetime'].max().date()} "
            f"({len(test_data):,} jogos)"
        )
        model = train_calibrated_xgboost_model(
            model_train.loc[:, feature_cols],
            model_train[RESULT_TARGET_COL],
            config.calibration_size,
            config.calibration_method,
            config.xgb_tuning_trials,
            config.xgb_tuning_validation_size,
            sample_weight=_walk_forward_sample_weights(model_train, config),
        )
        probabilities = model.predict_proba(test_data.loc[:, feature_cols])
        alpha_model = 1.0
        blend_validation_rows = 0
        if not blend_validation.empty:
            validation_probabilities = model.predict_proba(
                blend_validation.loc[:, feature_cols]
            )
            probabilities, alpha_model, blend_validation_rows = (
                _blend_fold_probabilities(
                    blend_validation[RESULT_TARGET_COL],
                    validation_probabilities,
                    probabilities,
                    blend_validation,
                    test_data,
                    [NO_VIG_HOME_COL, NO_VIG_DRAW_COL, NO_VIG_AWAY_COL],
                    [0, 1, 2],
                    "Resultado 1X2",
                )
            )

        metrics = evaluate_match_result_model(
            test_data[RESULT_TARGET_COL],
            probabilities,
        )
        _, backtest_summary = run_match_result_backtest(
            test_data,
            probabilities,
            stake=config.stake,
            edge=config.edge,
            min_model_prob=config.min_result_prob,
            max_result_odd=config.max_result_odd,
        )
        row = _base_walk_forward_row(
            "Resultado 1X2",
            fold,
            train_data,
            test_data,
            alpha_model,
            blend_validation_rows,
        )
        row.update(
            {
                "accuracy": metrics["accuracy"],
                "precision_h": metrics["precision_h"],
                "precision_d": metrics["precision_d"],
                "precision_a": metrics["precision_a"],
                "logloss": metrics["logloss"],
                "brier_score": metrics["brier_score"],
                "calibration_ece": metrics["calibration_ece"],
                **backtest_summary,
            }
        )
        fold_rows.append(row)

    summary = pd.DataFrame(fold_rows)
    if not summary.empty:
        _print_walk_forward_summary(summary, "1X2")
    return summary


def run_win_walk_forward_validation(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    config: PipelineConfig,
    initial_train_fraction: float | None = None,
) -> pd.DataFrame:
    """Executa validacao walk-forward por blocos de datas para vitorias."""
    if config.walk_forward_splits <= 0:
        return pd.DataFrame()

    train_fraction = (
        config.walk_forward_initial_train_fraction
        if initial_train_fraction is None
        else initial_train_fraction
    )
    folds = _build_walk_forward_date_folds(data, config, train_fraction)
    fold_rows: list[dict[str, float | int | str]] = []
    print("\n========== WALK-FORWARD VALIDATION VITORIA ==========")

    for fold, train_data, test_data in folds:
        if not _has_required_classes(train_data[RESULT_TARGET_COL], [0, 1, 2]):
            print(f"[walk-forward Vitoria] Fold {fold} ignorado: treino sem 3 classes.")
            continue

        model_train, blend_validation = _split_walk_forward_model_train(
            train_data,
            RESULT_TARGET_COL,
            config,
            [0, 1, 2],
        )
        print(
            f"[walk-forward Vitoria] Fold {fold}/{len(folds)}: "
            f"treino ate {train_data['MatchDatetime'].max().date()}, "
            f"teste {test_data['MatchDatetime'].min().date()} a "
            f"{test_data['MatchDatetime'].max().date()} "
            f"({len(test_data):,} jogos)"
        )
        model = train_calibrated_xgboost_model(
            model_train.loc[:, feature_cols],
            model_train[RESULT_TARGET_COL],
            config.calibration_size,
            config.calibration_method,
            config.xgb_tuning_trials,
            config.xgb_tuning_validation_size,
            sample_weight=_walk_forward_sample_weights(model_train, config),
        )
        probabilities = model.predict_proba(test_data.loc[:, feature_cols])
        alpha_model = 1.0
        blend_validation_rows = 0
        if not blend_validation.empty:
            validation_probabilities = model.predict_proba(
                blend_validation.loc[:, feature_cols]
            )
            probabilities, alpha_model, blend_validation_rows = (
                _blend_fold_probabilities(
                    blend_validation[RESULT_TARGET_COL],
                    validation_probabilities,
                    probabilities,
                    blend_validation,
                    test_data,
                    [NO_VIG_HOME_COL, NO_VIG_DRAW_COL, NO_VIG_AWAY_COL],
                    [0, 1, 2],
                    "Vitoria Casa/Fora",
                )
            )

        metrics = evaluate_match_result_model(
            test_data[RESULT_TARGET_COL],
            probabilities,
        )
        _, backtest_summary = run_win_backtest(
            test_data,
            probabilities,
            stake=config.stake,
            edge=config.edge,
            min_model_prob=config.min_win_prob,
            max_win_odd=config.max_win_odd,
        )
        row = _base_walk_forward_row(
            "Vitoria Casa/Fora",
            fold,
            train_data,
            test_data,
            alpha_model,
            blend_validation_rows,
        )
        row.update(
            {
                "accuracy": metrics["accuracy"],
                "precision_h": metrics["precision_h"],
                "precision_a": metrics["precision_a"],
                "logloss": metrics["logloss"],
                "brier_score": metrics["brier_score"],
                "calibration_ece": metrics["calibration_ece"],
                **backtest_summary,
            }
        )
        fold_rows.append(row)

    summary = pd.DataFrame(fold_rows)
    if not summary.empty:
        _print_walk_forward_summary(summary, "VITORIA")
    return summary


def summarize_walk_forward_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Resume estabilidade temporal por mercado."""
    if summary.empty or "Market" not in summary.columns:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for market, market_data in summary.groupby("Market", sort=False):
        market_data = market_data.copy()
        total_bets = float(market_data["bets"].sum())
        total_staked = float(market_data["total_staked"].sum())
        total_profit = float(market_data["total_profit"].sum())
        aggregate_roi = total_profit / total_staked if total_staked > 0 else 0.0
        positive_roi_rate = float(market_data["roi"].gt(0).mean())
        weighted_hit_rate = (
            float((market_data["hit_rate"] * market_data["bets"]).sum() / total_bets)
            if total_bets > 0
            else 0.0
        )
        roi_std = (
            float(market_data["roi"].std(ddof=0)) if len(market_data) > 1 else 0.0
        )
        stability_score = aggregate_roi - (roi_std * 0.50)
        if total_bets < 30:
            status = "Pouco volume"
        elif aggregate_roi > 0 and positive_roi_rate >= 0.60:
            status = "Estavel"
        elif aggregate_roi > 0:
            status = "Promissor"
        else:
            status = "Instavel"

        rows.append(
            {
                "Market": market,
                "Folds": len(market_data),
                "PositiveROIFolds": int(market_data["roi"].gt(0).sum()),
                "PositiveROIRate": positive_roi_rate,
                "MeanROI": float(market_data["roi"].mean()),
                "MedianROI": float(market_data["roi"].median()),
                "StdROI": roi_std,
                "MinROI": float(market_data["roi"].min()),
                "MaxROI": float(market_data["roi"].max()),
                "AggregateROI": aggregate_roi,
                "TotalBets": total_bets,
                "TotalStaked": total_staked,
                "TotalProfit": total_profit,
                "WeightedHitRate": weighted_hit_rate,
                "MeanLogLoss": float(market_data["logloss"].mean()),
                "StdLogLoss": (
                    float(market_data["logloss"].std(ddof=0))
                    if len(market_data) > 1
                    else 0.0
                ),
                "MeanBrierScore": float(market_data["brier_score"].mean()),
                "MeanCalibrationECE": float(
                    market_data["calibration_ece"].mean()
                ),
                "MeanAlphaModel": float(market_data["alpha_model"].mean()),
                "WorstFoldProfit": float(market_data["total_profit"].min()),
                "BestFoldProfit": float(market_data["total_profit"].max()),
                "StabilityScore": stability_score,
                "Status": status,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["Status", "StabilityScore"],
        ascending=[True, False],
        kind="mergesort",
    )
