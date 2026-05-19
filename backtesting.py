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
    evaluate_match_result_model,
    evaluate_model,
    train_calibrated_xgboost_model,
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


def run_walk_forward_validation(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    config: PipelineConfig,
    initial_train_fraction: float = 0.50,
) -> pd.DataFrame:
    """Executa validacao walk-forward com janelas expansivas."""
    if config.walk_forward_splits <= 0:
        return pd.DataFrame()

    if not 0.2 <= initial_train_fraction < 1.0:
        raise ValueError("initial_train_fraction deve estar entre 0.2 e 1.")

    data = data.sort_values("MatchDatetime", kind="mergesort").reset_index(drop=True)
    initial_train_size = int(len(data) * initial_train_fraction)
    remaining_size = len(data) - initial_train_size
    test_size = remaining_size // config.walk_forward_splits

    if initial_train_size <= 0 or test_size <= 0:
        raise ValueError("Dados insuficientes para walk-forward validation.")

    fold_rows: list[dict[str, float | int | str]] = []
    print("\n========== WALK-FORWARD VALIDATION ==========")

    for fold in range(1, config.walk_forward_splits + 1):
        train_end = initial_train_size + ((fold - 1) * test_size)
        test_end = (
            initial_train_size + (fold * test_size)
            if fold < config.walk_forward_splits
            else len(data)
        )

        train_data = data.iloc[:train_end].copy()
        test_data = data.iloc[train_end:test_end].copy()
        x_train = train_data.loc[:, feature_cols]
        y_train = train_data[TARGET_COL]
        x_test = test_data.loc[:, feature_cols]
        y_test = test_data[TARGET_COL]

        print(
            f"[walk-forward] Fold {fold}/{config.walk_forward_splits}: "
            f"treino {len(train_data):,}, teste {len(test_data):,}"
        )
        model = train_calibrated_xgboost_model(
            x_train,
            y_train,
            config.calibration_size,
            config.calibration_method,
            config.xgb_tuning_trials,
            config.xgb_tuning_validation_size,
        )
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics = evaluate_model(y_test, probabilities)
        _, backtest_summary = run_backtest(
            test_data,
            probabilities,
            stake=config.stake,
            edge=config.edge,
            min_model_prob=config.min_model_prob,
            max_over_odd=config.max_over_odd,
        )

        fold_rows.append(
            {
                "fold": fold,
                "train_start": train_data["MatchDatetime"].min().date().isoformat(),
                "train_end": train_data["MatchDatetime"].max().date().isoformat(),
                "test_start": test_data["MatchDatetime"].min().date().isoformat(),
                "test_end": test_data["MatchDatetime"].max().date().isoformat(),
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                "accuracy": metrics["accuracy"],
                "precision_over": metrics["precision_over"],
                "logloss": metrics["logloss"],
                "brier_score": metrics["brier_score"],
                "calibration_ece": metrics["calibration_ece"],
                "bets": backtest_summary["bets"],
                "total_staked": backtest_summary["total_staked"],
                "total_profit": backtest_summary["total_profit"],
                "roi": backtest_summary["roi"],
                "hit_rate": backtest_summary["hit_rate"],
                "avg_edge": backtest_summary["avg_edge"],
                "avg_model_prob": backtest_summary["avg_model_prob"],
                "avg_odd": backtest_summary["avg_odd"],
            }
        )

    summary = pd.DataFrame(fold_rows)
    total_staked = float(summary["total_staked"].sum())
    total_profit = float(summary["total_profit"].sum())
    total_bets = float(summary["bets"].sum())
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    weighted_hit_rate = (
        float((summary["hit_rate"] * summary["bets"]).sum() / total_bets)
        if total_bets > 0
        else 0.0
    )

    print("\n========== RESUMO WALK-FORWARD ==========")
    print(f"Folds:                    {len(summary)}")
    print(f"Acuracia media:           {summary['accuracy'].mean():.2%}")
    print(f"LogLoss medio:            {summary['logloss'].mean():.4f}")
    print(f"Brier medio:              {summary['brier_score'].mean():.4f}")
    print(f"ECE medio:                {summary['calibration_ece'].mean():.2%}")
    print(f"Apostas simuladas:        {summary['bets'].sum():.0f}")
    print(f"Taxa de acerto apostas:   {weighted_hit_rate:.2%}")
    print(f"Total apostado:           R$ {total_staked:.2f}")
    print(f"Lucro/Prejuizo:           R$ {total_profit:.2f}")
    print(f"ROI agregado:             {roi:.2%}")

    return summary


def run_match_result_walk_forward_validation(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    config: PipelineConfig,
    initial_train_fraction: float = 0.50,
) -> pd.DataFrame:
    """Executa validacao walk-forward para Resultado Final 1X2."""
    if config.walk_forward_splits <= 0:
        return pd.DataFrame()

    data = data.sort_values("MatchDatetime", kind="mergesort").reset_index(drop=True)
    initial_train_size = int(len(data) * initial_train_fraction)
    remaining_size = len(data) - initial_train_size
    test_size = remaining_size // config.walk_forward_splits

    if initial_train_size <= 0 or test_size <= 0:
        raise ValueError("Dados insuficientes para walk-forward validation 1X2.")

    fold_rows: list[dict[str, float | int | str]] = []
    print("\n========== WALK-FORWARD VALIDATION 1X2 ==========")

    for fold in range(1, config.walk_forward_splits + 1):
        train_end = initial_train_size + ((fold - 1) * test_size)
        test_end = (
            initial_train_size + (fold * test_size)
            if fold < config.walk_forward_splits
            else len(data)
        )

        train_data = data.iloc[:train_end].copy()
        test_data = data.iloc[train_end:test_end].copy()
        x_train = train_data.loc[:, feature_cols]
        y_train = train_data[RESULT_TARGET_COL]
        x_test = test_data.loc[:, feature_cols]
        y_test = test_data[RESULT_TARGET_COL]

        print(
            f"[walk-forward 1X2] Fold {fold}/{config.walk_forward_splits}: "
            f"treino {len(train_data):,}, teste {len(test_data):,}"
        )
        model = train_calibrated_xgboost_model(
            x_train,
            y_train,
            config.calibration_size,
            config.calibration_method,
            config.xgb_tuning_trials,
            config.xgb_tuning_validation_size,
        )
        probabilities = model.predict_proba(x_test)
        metrics = evaluate_match_result_model(y_test, probabilities)
        _, backtest_summary = run_match_result_backtest(
            test_data,
            probabilities,
            stake=config.stake,
            edge=config.edge,
            min_model_prob=config.min_result_prob,
            max_result_odd=config.max_result_odd,
        )

        fold_rows.append(
            {
                "fold": fold,
                "train_start": train_data["MatchDatetime"].min().date().isoformat(),
                "train_end": train_data["MatchDatetime"].max().date().isoformat(),
                "test_start": test_data["MatchDatetime"].min().date().isoformat(),
                "test_end": test_data["MatchDatetime"].max().date().isoformat(),
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                "accuracy": metrics["accuracy"],
                "precision_h": metrics["precision_h"],
                "precision_d": metrics["precision_d"],
                "precision_a": metrics["precision_a"],
                "logloss": metrics["logloss"],
                "brier_score": metrics["brier_score"],
                "calibration_ece": metrics["calibration_ece"],
                "bets": backtest_summary["bets"],
                "total_staked": backtest_summary["total_staked"],
                "total_profit": backtest_summary["total_profit"],
                "roi": backtest_summary["roi"],
                "hit_rate": backtest_summary["hit_rate"],
                "avg_edge": backtest_summary["avg_edge"],
                "avg_model_prob": backtest_summary["avg_model_prob"],
                "avg_odd": backtest_summary["avg_odd"],
            }
        )

    summary = pd.DataFrame(fold_rows)
    total_staked = float(summary["total_staked"].sum())
    total_profit = float(summary["total_profit"].sum())
    total_bets = float(summary["bets"].sum())
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    weighted_hit_rate = (
        float((summary["hit_rate"] * summary["bets"]).sum() / total_bets)
        if total_bets > 0
        else 0.0
    )

    print("\n========== RESUMO WALK-FORWARD 1X2 ==========")
    print(f"Folds:                    {len(summary)}")
    print(f"Acuracia media:           {summary['accuracy'].mean():.2%}")
    print(f"LogLoss medio:            {summary['logloss'].mean():.4f}")
    print(f"Brier medio:              {summary['brier_score'].mean():.4f}")
    print(f"ECE medio:                {summary['calibration_ece'].mean():.2%}")
    print(f"Apostas simuladas:        {summary['bets'].sum():.0f}")
    print(f"Taxa de acerto apostas:   {weighted_hit_rate:.2%}")
    print(f"Total apostado:           R$ {total_staked:.2f}")
    print(f"Lucro/Prejuizo:           R$ {total_profit:.2f}")
    print(f"ROI agregado:             {roi:.2%}")

    return summary


def run_win_walk_forward_validation(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    config: PipelineConfig,
    initial_train_fraction: float = 0.50,
) -> pd.DataFrame:
    """Executa validacao walk-forward para Vitoria Casa/Fora."""
    if config.walk_forward_splits <= 0:
        return pd.DataFrame()

    data = data.sort_values("MatchDatetime", kind="mergesort").reset_index(drop=True)
    initial_train_size = int(len(data) * initial_train_fraction)
    remaining_size = len(data) - initial_train_size
    test_size = remaining_size // config.walk_forward_splits

    if initial_train_size <= 0 or test_size <= 0:
        raise ValueError("Dados insuficientes para walk-forward validation vitoria.")

    fold_rows: list[dict[str, float | int | str]] = []
    print("\n========== WALK-FORWARD VALIDATION VITORIA ==========")

    for fold in range(1, config.walk_forward_splits + 1):
        train_end = initial_train_size + ((fold - 1) * test_size)
        test_end = (
            initial_train_size + (fold * test_size)
            if fold < config.walk_forward_splits
            else len(data)
        )

        train_data = data.iloc[:train_end].copy()
        test_data = data.iloc[train_end:test_end].copy()
        x_train = train_data.loc[:, feature_cols]
        y_train = train_data[RESULT_TARGET_COL]
        x_test = test_data.loc[:, feature_cols]
        y_test = test_data[RESULT_TARGET_COL]

        print(
            f"[walk-forward vitoria] Fold {fold}/{config.walk_forward_splits}: "
            f"treino {len(train_data):,}, teste {len(test_data):,}"
        )
        model = train_calibrated_xgboost_model(
            x_train,
            y_train,
            config.calibration_size,
            config.calibration_method,
            config.xgb_tuning_trials,
            config.xgb_tuning_validation_size,
        )
        probabilities = model.predict_proba(x_test)
        metrics = evaluate_match_result_model(y_test, probabilities)
        _, backtest_summary = run_win_backtest(
            test_data,
            probabilities,
            stake=config.stake,
            edge=config.edge,
            min_model_prob=config.min_win_prob,
            max_win_odd=config.max_win_odd,
        )

        fold_rows.append(
            {
                "fold": fold,
                "train_start": train_data["MatchDatetime"].min().date().isoformat(),
                "train_end": train_data["MatchDatetime"].max().date().isoformat(),
                "test_start": test_data["MatchDatetime"].min().date().isoformat(),
                "test_end": test_data["MatchDatetime"].max().date().isoformat(),
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                "accuracy": metrics["accuracy"],
                "precision_h": metrics["precision_h"],
                "precision_a": metrics["precision_a"],
                "logloss": metrics["logloss"],
                "brier_score": metrics["brier_score"],
                "calibration_ece": metrics["calibration_ece"],
                "bets": backtest_summary["bets"],
                "total_staked": backtest_summary["total_staked"],
                "total_profit": backtest_summary["total_profit"],
                "roi": backtest_summary["roi"],
                "hit_rate": backtest_summary["hit_rate"],
                "avg_edge": backtest_summary["avg_edge"],
                "avg_model_prob": backtest_summary["avg_model_prob"],
                "avg_odd": backtest_summary["avg_odd"],
            }
        )

    summary = pd.DataFrame(fold_rows)
    total_staked = float(summary["total_staked"].sum())
    total_profit = float(summary["total_profit"].sum())
    total_bets = float(summary["bets"].sum())
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    weighted_hit_rate = (
        float((summary["hit_rate"] * summary["bets"]).sum() / total_bets)
        if total_bets > 0
        else 0.0
    )

    print("\n========== RESUMO WALK-FORWARD VITORIA ==========")
    print(f"Folds:                    {len(summary)}")
    print(f"Acuracia 1X2 media:       {summary['accuracy'].mean():.2%}")
    print(f"LogLoss 1X2 medio:        {summary['logloss'].mean():.4f}")
    print(f"Brier 1X2 medio:          {summary['brier_score'].mean():.4f}")
    print(f"ECE medio:                {summary['calibration_ece'].mean():.2%}")
    print(f"Apostas simuladas:        {summary['bets'].sum():.0f}")
    print(f"Taxa de acerto apostas:   {weighted_hit_rate:.2%}")
    print(f"Total apostado:           R$ {total_staked:.2f}")
    print(f"Lucro/Prejuizo:           R$ {total_profit:.2f}")
    print(f"ROI agregado:             {roi:.2%}")

    return summary
