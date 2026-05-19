"""Backtest realista com validacao por temporada e gestao de stake."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from config import PipelineConfig
from filter_optimizer import MARKET_COLUMNS, _market_grid


STAKE_MODE_NAMES = {
    "fixed": "Stake fixa",
    "kelly_fractional": "Kelly fracionado",
}


@dataclass(frozen=True)
class SeasonSplit:
    """Conjuntos de treino, validacao e teste separados por temporada."""

    train_data: pd.DataFrame
    validation_data: pd.DataFrame
    test_data: pd.DataFrame
    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    train_seasons: list[str]
    validation_seasons: list[str]
    test_seasons: list[str]


def _season_order(data: pd.DataFrame) -> list[str]:
    """Ordena temporadas pela primeira partida disponivel."""
    seasons = (
        data.dropna(subset=["Temporada", "MatchDatetime"])
        .assign(Temporada=lambda frame: frame["Temporada"].astype(str))
        .groupby("Temporada", as_index=False)
        .agg(SeasonStart=("MatchDatetime", "min"), Rows=("Temporada", "size"))
        .sort_values(["SeasonStart", "Temporada"], kind="mergesort")
    )
    return seasons["Temporada"].tolist()


def split_train_validation_test_by_season(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    validation_seasons: int,
    test_seasons: int,
) -> SeasonSplit:
    """Separa treino/validacao/teste usando temporadas inteiras por liga."""
    if validation_seasons <= 0 or test_seasons <= 0:
        raise ValueError("validation_seasons e test_seasons devem ser positivos.")

    train_frames = []
    validation_frames = []
    test_frames = []
    train_season_labels: set[str] = set()
    validation_season_labels: set[str] = set()
    test_season_labels: set[str] = set()
    required_seasons = validation_seasons + test_seasons + 1

    for league, league_data in data.groupby("Liga", sort=False):
        seasons = _season_order(league_data)
        if len(seasons) < required_seasons:
            print(
                "[split] Liga ignorada por poucas temporadas: "
                f"{league} ({len(seasons)} disponiveis)"
            )
            continue

        test_values = seasons[-test_seasons:]
        validation_values = seasons[
            -(validation_seasons + test_seasons) : -test_seasons
        ]
        train_values = seasons[: -(validation_seasons + test_seasons)]

        league_copy = league_data.copy()
        season_text = league_copy["Temporada"].astype(str)
        train_frames.append(league_copy[season_text.isin(train_values)])
        validation_frames.append(league_copy[season_text.isin(validation_values)])
        test_frames.append(league_copy[season_text.isin(test_values)])
        train_season_labels.update(f"{league}:{season}" for season in train_values)
        validation_season_labels.update(
            f"{league}:{season}" for season in validation_values
        )
        test_season_labels.update(f"{league}:{season}" for season in test_values)

    if not train_frames or not validation_frames or not test_frames:
        raise ValueError("Dados insuficientes para split por temporada.")

    train_data = pd.concat(train_frames, ignore_index=True)
    validation_data = pd.concat(validation_frames, ignore_index=True)
    test_data = pd.concat(test_frames, ignore_index=True)
    train_data = train_data.sort_values("MatchDatetime", kind="mergesort")
    validation_data = validation_data.sort_values("MatchDatetime", kind="mergesort")
    test_data = test_data.sort_values("MatchDatetime", kind="mergesort")

    if train_data.empty or validation_data.empty or test_data.empty:
        raise ValueError("Split por temporada gerou uma janela vazia.")

    print(
        "[split temporada] Treino: "
        f"{train_data['MatchDatetime'].min().date()} a "
        f"{train_data['MatchDatetime'].max().date()} "
        f"({len(train_data):,} jogos)"
    )
    print(
        "[split temporada] Validacao: "
        f"{validation_data['MatchDatetime'].min().date()} a "
        f"{validation_data['MatchDatetime'].max().date()} "
        f"({len(validation_data):,} jogos)"
    )
    print(
        "[split temporada] Teste: "
        f"{test_data['MatchDatetime'].min().date()} a "
        f"{test_data['MatchDatetime'].max().date()} "
        f"({len(test_data):,} jogos)"
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
        train_seasons=sorted(train_season_labels),
        validation_seasons=sorted(validation_season_labels),
        test_seasons=sorted(test_season_labels),
    )


def _hit_series(data: pd.DataFrame, market: str) -> pd.Series:
    """Calcula acerto de cada selecao simulada."""
    columns = MARKET_COLUMNS[market]
    target_col = columns["target_col"]
    prediction_col = columns["prediction_col"]

    if prediction_col is None:
        return data[target_col].eq(columns["positive_target"])
    return data[target_col].eq(data[prediction_col])


def _max_losing_streak(hits: Sequence[bool]) -> int:
    """Retorna a maior sequencia de reds entre apostas feitas."""
    longest = 0
    current = 0
    for hit in hits:
        if bool(hit):
            current = 0
            continue
        current += 1
        longest = max(longest, current)
    return longest


def _kelly_stake(
    bankroll: float,
    probability: float,
    odd: float,
    kelly_fraction: float,
    max_kelly_fraction: float,
) -> float:
    """Calcula stake pelo Kelly fracionado com teto de exposicao."""
    if bankroll <= 0 or odd <= 1.0:
        return 0.0

    payout = odd - 1.0
    full_kelly = ((payout * probability) - (1.0 - probability)) / payout
    stake_fraction = max(0.0, full_kelly) * kelly_fraction
    stake_fraction = min(stake_fraction, max_kelly_fraction)
    return float(bankroll * stake_fraction)


def simulate_threshold_strategy(
    data: pd.DataFrame,
    market: str,
    edge_threshold: float,
    min_model_prob: float,
    max_odd: float | None,
    stake_mode: str,
    fixed_stake: float,
    initial_bankroll: float,
    kelly_fraction: float,
    max_kelly_fraction: float,
) -> pd.DataFrame:
    """Simula uma estrategia em ordem cronologica."""
    columns = MARKET_COLUMNS[market]
    prob_col = columns["prob_col"]
    edge_col = columns["edge_col"]
    odd_col = columns["odd_col"]

    valid = data.dropna(subset=[prob_col, edge_col, odd_col]).copy()
    valid = valid.sort_values("MatchDatetime", kind="mergesort")
    odd_filter = True if max_odd is None else valid[odd_col].le(max_odd)
    bet_mask = valid[edge_col].ge(edge_threshold) & valid[prob_col].ge(
        min_model_prob
    ) & odd_filter
    bets = valid[bet_mask].copy()
    if bets.empty:
        return bets

    hits = _hit_series(bets, market).astype(bool).to_numpy()
    bankroll = float(initial_bankroll)
    rows = []
    for row_index, (_, row) in enumerate(bets.iterrows()):
        probability = float(row[prob_col])
        odd = float(row[odd_col])
        if stake_mode == "kelly_fractional":
            stake = _kelly_stake(
                bankroll,
                probability,
                odd,
                kelly_fraction,
                max_kelly_fraction,
            )
        else:
            stake = float(fixed_stake)

        if stake <= 0:
            continue

        profit = stake * (odd - 1.0) if hits[row_index] else -stake
        bankroll_before = bankroll
        bankroll += profit
        enriched = row.to_dict()
        enriched.update(
            {
                "Market": market,
                "StakeMode": stake_mode,
                "StakeModeName": STAKE_MODE_NAMES[stake_mode],
                "StrategyStake": stake,
                "StrategyProfit": profit,
                "StrategyHit": bool(hits[row_index]),
                "StrategyOdd": odd,
                "StrategyModelProb": probability,
                "StrategyEdge": float(row[edge_col]),
                "BankrollBefore": bankroll_before,
                "BankrollAfter": bankroll,
                "StrategyEdgeThreshold": edge_threshold,
                "StrategyMinModelProb": min_model_prob,
                "StrategyMaxOdd": np.nan if max_odd is None else max_odd,
            }
        )
        rows.append(enriched)

    simulated = pd.DataFrame(rows)
    if simulated.empty:
        return simulated

    simulated["CumulativeProfit"] = simulated["StrategyProfit"].cumsum()
    simulated["Equity"] = initial_bankroll + simulated["CumulativeProfit"]
    simulated["EquityPeak"] = simulated["Equity"].cummax()
    simulated["Drawdown"] = simulated["EquityPeak"] - simulated["Equity"]
    simulated["DrawdownPct"] = np.where(
        simulated["EquityPeak"].gt(0),
        simulated["Drawdown"] / simulated["EquityPeak"],
        0.0,
    )
    return simulated


def summarize_simulated_bets(
    bets: pd.DataFrame,
    initial_bankroll: float,
) -> dict[str, float]:
    """Resume retorno, drawdown e sequencia de reds de uma simulacao."""
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
            "MaxDrawdown": 0.0,
            "MaxDrawdownPct": 0.0,
            "LongestRedStreak": 0.0,
            "FinalBankroll": initial_bankroll,
        }

    columns = MARKET_COLUMNS[str(bets.iloc[0]["Market"])]
    total_staked = float(bets["StrategyStake"].sum())
    total_profit = float(bets["StrategyProfit"].sum())
    return {
        "Bets": float(len(bets)),
        "TotalStaked": total_staked,
        "TotalProfit": total_profit,
        "ROI": total_profit / total_staked if total_staked else 0.0,
        "HitRate": float(bets["StrategyHit"].mean()),
        "AvgEdge": float(bets[columns["edge_col"]].mean()),
        "AvgModelProb": float(bets[columns["prob_col"]].mean()),
        "AvgOdd": float(bets[columns["odd_col"]].mean()),
        "MaxDrawdown": float(bets["Drawdown"].max()),
        "MaxDrawdownPct": float(bets["DrawdownPct"].max()),
        "LongestRedStreak": float(_max_losing_streak(bets["StrategyHit"].tolist())),
        "FinalBankroll": float(initial_bankroll + total_profit),
    }


def _prefixed(prefix: str, summary: dict[str, float]) -> dict[str, float]:
    """Prefixa metricas de validacao ou teste."""
    return {f"{prefix}{key}": value for key, value in summary.items()}


def _period_bounds(data: pd.DataFrame) -> tuple[str, str]:
    """Retorna inicio e fim ISO de um periodo."""
    return (
        data["MatchDatetime"].min().date().isoformat(),
        data["MatchDatetime"].max().date().isoformat(),
    )


def _season_label(data: pd.DataFrame) -> str:
    """Resume temporadas presentes em um conjunto."""
    seasons = sorted(data["Temporada"].astype(str).dropna().unique().tolist())
    if len(seasons) <= 4:
        return ", ".join(seasons)
    return f"{seasons[0]} ... {seasons[-1]} ({len(seasons)})"


def build_threshold_grid(
    validation_backtest: pd.DataFrame,
    test_backtest: pd.DataFrame,
    market: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Testa multiplos thresholds em validacao e mede em teste final."""
    edge_grid, prob_grid, max_odd_grid = _market_grid(market)
    validation_start, validation_end = _period_bounds(validation_backtest)
    test_start, test_end = _period_bounds(test_backtest)
    rows = []

    for stake_mode in STAKE_MODE_NAMES:
        for edge_threshold in edge_grid:
            for min_model_prob in prob_grid:
                for max_odd in max_odd_grid:
                    validation_bets = simulate_threshold_strategy(
                        validation_backtest,
                        market,
                        edge_threshold,
                        min_model_prob,
                        max_odd,
                        stake_mode,
                        config.stake,
                        config.kelly_bankroll,
                        config.kelly_fraction,
                        config.max_kelly_fraction,
                    )
                    test_bets = simulate_threshold_strategy(
                        test_backtest,
                        market,
                        edge_threshold,
                        min_model_prob,
                        max_odd,
                        stake_mode,
                        config.stake,
                        config.kelly_bankroll,
                        config.kelly_fraction,
                        config.max_kelly_fraction,
                    )
                    validation_summary = summarize_simulated_bets(
                        validation_bets,
                        config.kelly_bankroll,
                    )
                    test_summary = summarize_simulated_bets(
                        test_bets,
                        config.kelly_bankroll,
                    )
                    rows.append(
                        {
                            "Market": market,
                            "StakeMode": stake_mode,
                            "StakeModeName": STAKE_MODE_NAMES[stake_mode],
                            "EdgeThreshold": edge_threshold,
                            "MinModelProb": min_model_prob,
                            "MaxOdd": np.nan if max_odd is None else max_odd,
                            "HasMaxOdd": max_odd is not None,
                            "ValidationStart": validation_start,
                            "ValidationEnd": validation_end,
                            "ValidationSeasons": _season_label(validation_backtest),
                            "TestStart": test_start,
                            "TestEnd": test_end,
                            "TestSeasons": _season_label(test_backtest),
                            "Qualified": (
                                validation_summary["Bets"]
                                >= config.min_optimization_bets
                            ),
                            **_prefixed("Val", validation_summary),
                            **_prefixed("Test", test_summary),
                        }
                    )

    return pd.DataFrame(rows)


def select_best_strategies(grid: pd.DataFrame) -> pd.DataFrame:
    """Escolhe uma estrategia por mercado e modo de stake pela validacao."""
    if grid.empty:
        return pd.DataFrame()

    qualified = grid[grid["Qualified"]].copy()
    if qualified.empty:
        qualified = grid[grid["ValBets"] > 0].copy()
    if qualified.empty:
        return pd.DataFrame()

    ranked = qualified.sort_values(
        ["Market", "StakeMode", "ValROI", "ValTotalProfit", "ValBets"],
        ascending=[True, True, False, False, False],
        kind="mergesort",
    )
    best = ranked.groupby(["Market", "StakeMode"], as_index=False).head(1).copy()
    best["MaxOddLabel"] = best["MaxOdd"].map(
        lambda value: "sem limite" if pd.isna(value) else f"{value:.2f}"
    )
    best["StrategyId"] = best.apply(
        lambda row: (
            f"{row['Market']} | {row['StakeModeName']} | "
            f"edge {row['EdgeThreshold']:.1%} | "
            f"prob {row['MinModelProb']:.1%} | "
            f"odd {row['MaxOddLabel']}"
        ),
        axis=1,
    )
    return best.reset_index(drop=True)


def _strategy_max_odd(row: pd.Series) -> float | None:
    """Converte a odd maxima de uma estrategia para None quando nao ha limite."""
    if "HasMaxOdd" in row.index and not bool(row["HasMaxOdd"]):
        return None
    if pd.isna(row["MaxOdd"]):
        return None
    return float(row["MaxOdd"])


def build_best_strategy_bets(
    test_backtest: pd.DataFrame,
    best_strategies: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Materializa as apostas de teste das melhores estrategias."""
    frames = []
    for _, strategy in best_strategies.iterrows():
        bets = simulate_threshold_strategy(
            test_backtest,
            str(strategy["Market"]),
            float(strategy["EdgeThreshold"]),
            float(strategy["MinModelProb"]),
            _strategy_max_odd(strategy),
            str(strategy["StakeMode"]),
            config.stake,
            config.kelly_bankroll,
            config.kelly_fraction,
            config.max_kelly_fraction,
        )
        if bets.empty:
            continue
        bets["StrategyId"] = strategy["StrategyId"]
        frames.append(bets)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summary_group(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Agrupa apostas simuladas e calcula retorno."""
    if data.empty:
        return pd.DataFrame()

    summary = (
        data.groupby(group_cols, as_index=False)
        .agg(
            Bets=("StrategyHit", "size"),
            TotalStaked=("StrategyStake", "sum"),
            TotalProfit=("StrategyProfit", "sum"),
            HitRate=("StrategyHit", "mean"),
            AvgOdd=("StrategyOdd", "mean"),
            AvgEdge=("StrategyEdge", "mean"),
            AvgModelProb=("StrategyModelProb", "mean"),
            MaxDrawdown=("Drawdown", "max"),
            LongestRedStreak=("StrategyHit", lambda values: _max_losing_streak(values)),
        )
    )
    summary["ROI"] = np.where(
        summary["TotalStaked"].gt(0),
        summary["TotalProfit"] / summary["TotalStaked"],
        0.0,
    )
    return summary


def build_monthly_breakdown(bets: pd.DataFrame) -> pd.DataFrame:
    """Resume lucro por mes para as estrategias escolhidas."""
    if bets.empty:
        return pd.DataFrame()

    data = bets.copy()
    data["Month"] = pd.to_datetime(data["MatchDatetime"]).dt.to_period("M").astype(str)
    return _summary_group(
        data,
        ["Market", "StakeMode", "StakeModeName", "StrategyId", "Month"],
    )


def build_league_breakdown(bets: pd.DataFrame) -> pd.DataFrame:
    """Resume lucro por liga para as estrategias escolhidas."""
    if bets.empty:
        return pd.DataFrame()

    group_cols = ["Market", "StakeMode", "StakeModeName", "StrategyId", "Liga"]
    if "LigaNome" in bets.columns:
        group_cols.append("LigaNome")
    return _summary_group(bets, group_cols)


def run_realistic_market_backtest(
    validation_backtest: pd.DataFrame,
    test_backtest: pd.DataFrame,
    market: str,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executa grid, escolhe estrategias e gera quebras de teste."""
    if validation_backtest.empty or test_backtest.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    grid = build_threshold_grid(validation_backtest, test_backtest, market, config)
    best = select_best_strategies(grid)
    bets = build_best_strategy_bets(test_backtest, best, config)
    monthly = build_monthly_breakdown(bets)
    league = build_league_breakdown(bets)
    return best, grid, bets, monthly, league
