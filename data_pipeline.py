"""Ingestao, limpeza e engenharia de features."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests

from config import (
    BASE_URL,
    EXTRA_LEAGUE_URLS,
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
    RAW_IMPLIED_AWAY_COL,
    RAW_IMPLIED_DRAW_COL,
    RAW_IMPLIED_HOME_COL,
    RAW_IMPLIED_OVER_COL,
    RAW_IMPLIED_UNDER_COL,
    RESULT_LABELS,
    RESULT_TARGET_COL,
    TARGET_COL,
)

OVER_CANDIDATES = [
    "B365>2.5",
    "B365C>2.5",
    "BbMx>2.5",
    "BbAv>2.5",
    "Max>2.5",
    "Avg>2.5",
    "MaxC>2.5",
    "AvgC>2.5",
    "P>2.5",
    "PC>2.5",
    "BFE>2.5",
    "BFEC>2.5",
]
UNDER_CANDIDATES = [
    "B365<2.5",
    "B365C<2.5",
    "BbMx<2.5",
    "BbAv<2.5",
    "Max<2.5",
    "Avg<2.5",
    "MaxC<2.5",
    "AvgC<2.5",
    "P<2.5",
    "PC<2.5",
    "BFE<2.5",
    "BFEC<2.5",
]
HOME_ODDS_CANDIDATES = [
    "B365H",
    "B365CH",
    "PSCH",
    "PSH",
    "BbMxH",
    "BbAvH",
    "MaxH",
    "MaxCH",
    "AvgH",
    "AvgCH",
]
DRAW_ODDS_CANDIDATES = [
    "B365D",
    "B365CD",
    "PSCD",
    "PSD",
    "BbMxD",
    "BbAvD",
    "MaxD",
    "MaxCD",
    "AvgD",
    "AvgCD",
]
AWAY_ODDS_CANDIDATES = [
    "B365A",
    "B365CA",
    "PSCA",
    "PSA",
    "BbMxA",
    "BbAvA",
    "MaxA",
    "MaxCA",
    "AvgA",
    "AvgCA",
]
OPEN_OVER_CANDIDATES = ["B365>2.5", "P>2.5", "Max>2.5", "Avg>2.5", "BFE>2.5"]
CLOSE_OVER_CANDIDATES = [
    "B365C>2.5",
    "PC>2.5",
    "MaxC>2.5",
    "AvgC>2.5",
    "BFEC>2.5",
]
OPEN_UNDER_CANDIDATES = ["B365<2.5", "P<2.5", "Max<2.5", "Avg<2.5", "BFE<2.5"]
CLOSE_UNDER_CANDIDATES = [
    "B365C<2.5",
    "PC<2.5",
    "MaxC<2.5",
    "AvgC<2.5",
    "BFEC<2.5",
]
OPEN_HOME_CANDIDATES = ["B365H", "PSH", "MaxH", "AvgH", "BFEH"]
CLOSE_HOME_CANDIDATES = ["B365CH", "PSCH", "MaxCH", "AvgCH", "BFECH"]
OPEN_DRAW_CANDIDATES = ["B365D", "PSD", "MaxD", "AvgD", "BFED"]
CLOSE_DRAW_CANDIDATES = ["B365CD", "PSCD", "MaxCD", "AvgCD", "BFECD"]
OPEN_AWAY_CANDIDATES = ["B365A", "PSA", "MaxA", "AvgA", "BFEA"]
CLOSE_AWAY_CANDIDATES = ["B365CA", "PSCA", "MaxCA", "AvgCA", "BFECA"]
MATCH_IMPORTANCE_FEATURE_COLS = [
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
    "Home_MatchImportance",
    "Away_MatchImportance",
    "MatchImportance",
    "Importance_Diff",
    "TopClash",
    "SeasonProgress",
]
LINEUP_FEATURE_COLS = [
    "Home_LineupDataAvailable",
    "Away_LineupDataAvailable",
    "Home_LineupConfirmed",
    "Away_LineupConfirmed",
    "Home_LineupStrength",
    "Away_LineupStrength",
    "LineupStrength_Diff",
    "Home_MissingStarters",
    "Away_MissingStarters",
    "MissingStarters_Diff",
    "Home_MissingKeyPlayers",
    "Away_MissingKeyPlayers",
    "MissingKeyPlayers_Diff",
]
TEAM_MATCH_STAT_METRICS = [
    "MatchStatsAvailable",
    "ShotsFor",
    "ShotsAgainst",
    "ShotsOnTargetFor",
    "ShotsOnTargetAgainst",
    "CornersFor",
    "CornersAgainst",
    "FoulsFor",
    "FoulsAgainst",
    "YellowCardsFor",
    "YellowCardsAgainst",
    "RedCardsFor",
    "RedCardsAgainst",
    "ShotAccuracy",
    "SOTAllowedRate",
    "ConversionRate",
    "GoalsAllowedPerShot",
    "FirstHalfGoalsFor",
    "FirstHalfGoalsAgainst",
    "FirstHalfTotalGoals",
    "SecondHalfGoalsFor",
    "SecondHalfGoalsAgainst",
    "SecondHalfTotalGoals",
]
VENUE_MATCH_STAT_METRICS = [
    "ShotsFor",
    "ShotsAgainst",
    "ShotsOnTargetFor",
    "ShotsOnTargetAgainst",
    "CornersFor",
    "CornersAgainst",
    "FirstHalfTotalGoals",
    "SecondHalfTotalGoals",
]
MATCH_STATS_DERIVED_FEATURE_COLS = [
    "Shots_Diff_Roll5",
    "ShotsAllowed_Diff_Roll5",
    "Shots_Total_Roll5",
    "ShotsOnTarget_Diff_Roll5",
    "ShotsOnTargetAllowed_Diff_Roll5",
    "ShotsOnTarget_Total_Roll5",
    "ShotPressure_Diff_Roll5",
    "ShotAccuracy_Diff_Roll5",
    "ConversionRate_Diff_Roll5",
    "Corners_Diff_Roll5",
    "Corners_Total_Roll5",
    "Discipline_Diff_Roll5",
    "RedCards_Diff_Roll5",
    "FirstHalfGoals_Total_Roll5",
    "SecondHalfGoals_Total_Roll5",
    "Venue_ShotsOnTarget_Total_Roll5",
    "Venue_Corners_Total_Roll5",
    "Venue_ShotPressure_Diff_Roll5",
]
MATCH_STATS_FEATURE_COLS = (
    [f"Home_{metric}_Roll5" for metric in TEAM_MATCH_STAT_METRICS]
    + [f"Away_{metric}_Roll5" for metric in TEAM_MATCH_STAT_METRICS]
    + [f"Home_Home_{metric}_Roll5" for metric in VENUE_MATCH_STAT_METRICS]
    + [f"Away_Away_{metric}_Roll5" for metric in VENUE_MATCH_STAT_METRICS]
    + MATCH_STATS_DERIVED_FEATURE_COLS
)
REFEREE_FEATURE_COLS = [
    "Referee_DataAvailable_Roll20",
    "Referee_TotalGoals_Roll20",
    "Referee_TotalCards_Roll20",
    "Referee_RedCards_Roll20",
    "Referee_Fouls_Roll20",
]
ODDS_MOVEMENT_FEATURE_COLS = [
    "Over25_ClosingAvailable",
    "Over25_OddsMove",
    "Over25_ImpliedMove",
    "Under25_OddsMove",
    "Under25_ImpliedMove",
    "Result_ClosingAvailable",
    "Home_OddsMove",
    "Home_ImpliedMove",
    "Draw_OddsMove",
    "Draw_ImpliedMove",
    "Away_OddsMove",
    "Away_ImpliedMove",
]
TEAM_XG_METRICS = [
    "xGAvailable",
    "xGFor",
    "xGAgainst",
    "xGTotal",
    "xGDiff",
    "FinishingDelta",
    "DefensiveDelta",
]
VENUE_XG_METRICS = [
    "xGFor",
    "xGAgainst",
    "xGTotal",
    "xGDiff",
]
XG_DERIVED_FEATURE_COLS = [
    "xG_Attack_Diff_Roll5",
    "xG_Defense_Diff_Roll5",
    "xG_Total_Roll5",
    "xG_Expected_Total_Match_Roll5",
    "xG_Diff_Roll5",
    "xG_Finishing_Diff_Roll5",
    "xG_Defensive_Delta_Diff_Roll5",
    "Venue_xG_Total_Roll5",
    "Venue_xG_Diff_Roll5",
]
XG_FEATURE_COLS = (
    [f"Home_{metric}_Roll5" for metric in TEAM_XG_METRICS]
    + [f"Away_{metric}_Roll5" for metric in TEAM_XG_METRICS]
    + [f"Home_Home_{metric}_Roll5" for metric in VENUE_XG_METRICS]
    + [f"Away_Away_{metric}_Roll5" for metric in VENUE_XG_METRICS]
    + XG_DERIVED_FEATURE_COLS
)
ELO_FEATURE_COLS = [
    "Home_Elo_Pre",
    "Away_Elo_Pre",
    "Elo_Diff",
    "Elo_Home_Adv_Diff",
    "Elo_Expected_Home",
    "Elo_Expected_Away",
]


def download_csv_if_needed(
    league: str,
    season: str,
    raw_dir: Path,
    timeout: int = 30,
) -> Path | None:
    """Baixa um CSV do football-data.co.uk se ele ainda nao existir.

    Args:
        league: Codigo da liga, por exemplo "E0" ou "SP1".
        season: Temporada no padrao do site, por exemplo "2425".
        raw_dir: Pasta onde o arquivo sera salvo.
        timeout: Timeout HTTP em segundos.

    Returns:
        Caminho local do CSV, ou None se o download falhar.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    if league in EXTRA_LEAGUE_URLS:
        file_path = raw_dir / f"{league}.csv"
        url = EXTRA_LEAGUE_URLS[league]
    else:
        file_path = raw_dir / f"{season}_{league}.csv"
        url = BASE_URL.format(season=season, league=league)

    if file_path.exists() and file_path.stat().st_size > 0:
        print(f"[cache] Usando arquivo local: {file_path}")
        return file_path

    print(f"[download] Baixando {url}")

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[aviso] Falha ao baixar {league} {season}: {exc}")
        return None

    file_path.write_bytes(response.content)
    time.sleep(1)
    return file_path


def load_football_data(
    leagues: Sequence[str],
    seasons: Sequence[str],
    raw_dir: Path,
) -> pd.DataFrame:
    """Baixa e concatena os CSVs solicitados em um DataFrame unificado."""
    frames: list[pd.DataFrame] = []

    for league in leagues:
        if league in EXTRA_LEAGUE_URLS:
            file_path = download_csv_if_needed(league, "all", raw_dir)
            if file_path is None:
                continue

            data = pd.read_csv(file_path, encoding="utf-8-sig")
            data.columns = data.columns.str.strip()
            data = data.rename(
                columns={
                    "Home": "HomeTeam",
                    "Away": "AwayTeam",
                    "HG": "FTHG",
                    "AG": "FTAG",
                    "Res": "FTR",
                }
            )
            data["Liga"] = league
            data["Temporada"] = data["Season"].astype(str)
            frames.append(data.copy())
            continue

        for season in seasons:
            file_path = download_csv_if_needed(league, season, raw_dir)
            if file_path is None:
                continue

            data = pd.read_csv(file_path, encoding="utf-8-sig")
            data.columns = data.columns.str.strip()
            metadata = pd.DataFrame(
                {
                    "Liga": league,
                    "Temporada": season,
                },
                index=data.index,
            )
            data = pd.concat([data, metadata], axis=1).copy()
            frames.append(data)

    if not frames:
        raise RuntimeError(
            "Nenhum CSV valido foi carregado. Confira ligas, temporadas e "
            "conexao com a internet."
        )

    return pd.concat(frames, ignore_index=True)


def parse_match_datetime(data: pd.DataFrame) -> pd.DataFrame:
    """Cria uma coluna MatchDatetime para ordenacao cronologica estrita."""
    if "Date" not in data.columns:
        raise KeyError("A coluna obrigatoria 'Date' nao foi encontrada.")

    data = data.copy()
    date_text = data["Date"].astype(str).str.strip()

    if "Time" in data.columns:
        time_text = data["Time"].fillna("00:00").astype(str).str.strip()
        time_text = time_text.mask(time_text.eq(""), "00:00")
    else:
        time_text = pd.Series("00:00", index=data.index)

    combined_datetime = date_text + " " + time_text
    parsed_datetime = pd.to_datetime(
        combined_datetime,
        dayfirst=True,
        errors="coerce",
    )
    parsed_date = pd.to_datetime(date_text, dayfirst=True, errors="coerce")

    data["MatchDatetime"] = parsed_datetime.fillna(parsed_date)
    return data


def resolve_over_under_odds_columns(data: pd.DataFrame) -> tuple[str, str]:
    """Resolve colunas de odds Over/Under 2.5 com fallback robusto.

    A prioridade e Bet365 pre-closing, depois Bet365 closing, e em seguida
    colunas agregadas de mercado historicas/atuais do football-data.
    """
    over_col = next((col for col in OVER_CANDIDATES if col in data.columns), None)
    under_col = next((col for col in UNDER_CANDIDATES if col in data.columns), None)

    if over_col is None or under_col is None:
        raise KeyError(
            "Nao encontrei colunas de odds Over/Under 2.5. Colunas "
            f"disponiveis: {list(data.columns)}"
        )

    return over_col, under_col


def resolve_match_result_odds_columns(data: pd.DataFrame) -> tuple[str, str, str]:
    """Resolve colunas de odds 1X2 com fallback robusto."""
    home_col = next((col for col in HOME_ODDS_CANDIDATES if col in data.columns), None)
    draw_col = next((col for col in DRAW_ODDS_CANDIDATES if col in data.columns), None)
    away_col = next((col for col in AWAY_ODDS_CANDIDATES if col in data.columns), None)

    if home_col is None or draw_col is None or away_col is None:
        raise KeyError(
            "Nao encontrei colunas de odds 1X2. Colunas disponiveis: "
            f"{list(data.columns)}"
        )

    return home_col, draw_col, away_col


def coalesce_numeric_columns(
    data: pd.DataFrame,
    candidates: Sequence[str],
) -> tuple[pd.Series, str]:
    """Combina colunas candidatas linha a linha seguindo ordem de prioridade."""
    available_cols = [col for col in candidates if col in data.columns]
    if not available_cols:
        return pd.Series(np.nan, index=data.index), ""

    converted = pd.DataFrame(index=data.index)
    for col in available_cols:
        converted[col] = pd.to_numeric(data[col], errors="coerce")

    return converted.bfill(axis=1).iloc[:, 0], available_cols[0]


def prepare_initial_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, str, str, str]:
    """Ordena, padroniza tipos numericos e remove jogos incompletos."""
    required_cols = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Colunas obrigatorias ausentes: {missing_cols}")

    data = parse_match_datetime(data)
    try:
        over_col, under_col = resolve_over_under_odds_columns(data)
    except KeyError:
        over_col, under_col = "", ""
    home_col, draw_col, away_col = resolve_match_result_odds_columns(data)

    numeric_cols = ["FTHG", "FTAG"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(
        subset=[
            "MatchDatetime",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
        ]
    ).copy()

    data[ODD_OVER_COL], _ = coalesce_numeric_columns(data, OVER_CANDIDATES)
    data[ODD_UNDER_COL], _ = coalesce_numeric_columns(data, UNDER_CANDIDATES)
    data[ODD_HOME_COL], _ = coalesce_numeric_columns(data, HOME_ODDS_CANDIDATES)
    data[ODD_DRAW_COL], _ = coalesce_numeric_columns(data, DRAW_ODDS_CANDIDATES)
    data[ODD_AWAY_COL], _ = coalesce_numeric_columns(data, AWAY_ODDS_CANDIDATES)
    data = add_no_vig_market_probabilities(data)
    data = add_odds_movement_features(data)

    data = data.sort_values(
        ["MatchDatetime", "Liga", "Temporada", "HomeTeam", "AwayTeam"],
        kind="mergesort",
    ).reset_index(drop=True)

    if over_col and under_col:
        print(f"[odds] Over 2.5 usando coluna: {over_col}")
        print(f"[odds] Under 2.5 usando coluna: {under_col}")
    else:
        print("[odds] Over/Under 2.5 indisponivel para parte da base.")
    print(
        "[odds] 1X2 usando colunas: "
        f"{home_col}/{draw_col}/{away_col}"
    )
    print(f"[dados] Jogos completos apos limpeza inicial: {len(data):,}")

    return data, over_col, under_col, home_col, draw_col, away_col


def add_no_vig_market_probabilities(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula probabilidades implicitas cruas e sem margem da casa."""
    data = data.copy()
    data[RAW_IMPLIED_OVER_COL] = 1.0 / data[ODD_OVER_COL]
    data[RAW_IMPLIED_UNDER_COL] = 1.0 / data[ODD_UNDER_COL]
    data[OVERROUND_COL] = (
        data[RAW_IMPLIED_OVER_COL] + data[RAW_IMPLIED_UNDER_COL]
    )
    data[NO_VIG_OVER_COL] = data[RAW_IMPLIED_OVER_COL] / data[OVERROUND_COL]
    data[NO_VIG_UNDER_COL] = data[RAW_IMPLIED_UNDER_COL] / data[OVERROUND_COL]

    data[RAW_IMPLIED_HOME_COL] = 1.0 / data[ODD_HOME_COL]
    data[RAW_IMPLIED_DRAW_COL] = 1.0 / data[ODD_DRAW_COL]
    data[RAW_IMPLIED_AWAY_COL] = 1.0 / data[ODD_AWAY_COL]
    data[OVERROUND_1X2_COL] = (
        data[RAW_IMPLIED_HOME_COL]
        + data[RAW_IMPLIED_DRAW_COL]
        + data[RAW_IMPLIED_AWAY_COL]
    )
    data[NO_VIG_HOME_COL] = data[RAW_IMPLIED_HOME_COL] / data[OVERROUND_1X2_COL]
    data[NO_VIG_DRAW_COL] = data[RAW_IMPLIED_DRAW_COL] / data[OVERROUND_1X2_COL]
    data[NO_VIG_AWAY_COL] = data[RAW_IMPLIED_AWAY_COL] / data[OVERROUND_1X2_COL]
    return data


def _valid_odds_mask(*series: pd.Series) -> pd.Series:
    """Indica linhas em que todas as odds sao validas."""
    mask = pd.Series(True, index=series[0].index)
    for values in series:
        numeric_values = pd.to_numeric(values, errors="coerce")
        mask &= numeric_values.gt(1.0) & np.isfinite(numeric_values)
    return mask


def _add_two_way_odds_move(
    data: pd.DataFrame,
    opening_candidates: Sequence[str],
    closing_candidates: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    """Adiciona movimento de odds para uma selecao."""
    opening, _ = coalesce_numeric_columns(data, opening_candidates)
    closing, _ = coalesce_numeric_columns(data, closing_candidates)
    valid_mask = _valid_odds_mask(opening, closing)
    odds_move = (closing - opening).where(valid_mask, 0.0)
    implied_move = ((1.0 / closing) - (1.0 / opening)).where(valid_mask, 0.0)

    data[f"{prefix}_OddsMove"] = odds_move.fillna(0.0)
    data[f"{prefix}_ImpliedMove"] = implied_move.fillna(0.0)
    return data


def add_odds_movement_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria features de movimento entre odds de abertura e fechamento."""
    data = data.copy()

    opening_over, _ = coalesce_numeric_columns(data, OPEN_OVER_CANDIDATES)
    closing_over, _ = coalesce_numeric_columns(data, CLOSE_OVER_CANDIDATES)
    over_valid = _valid_odds_mask(opening_over, closing_over)
    data["Over25_ClosingAvailable"] = over_valid.astype(float)
    data["Over25_OddsMove"] = (closing_over - opening_over).where(
        over_valid,
        0.0,
    ).fillna(0.0)
    data["Over25_ImpliedMove"] = (
        (1.0 / closing_over) - (1.0 / opening_over)
    ).where(over_valid, 0.0).fillna(0.0)

    opening_under, _ = coalesce_numeric_columns(data, OPEN_UNDER_CANDIDATES)
    closing_under, _ = coalesce_numeric_columns(data, CLOSE_UNDER_CANDIDATES)
    under_valid = _valid_odds_mask(opening_under, closing_under)
    data["Under25_OddsMove"] = (closing_under - opening_under).where(
        under_valid,
        0.0,
    ).fillna(0.0)
    data["Under25_ImpliedMove"] = (
        (1.0 / closing_under) - (1.0 / opening_under)
    ).where(under_valid, 0.0).fillna(0.0)

    result_valid_frames = []
    for prefix, opening_cols, closing_cols in [
        ("Home", OPEN_HOME_CANDIDATES, CLOSE_HOME_CANDIDATES),
        ("Draw", OPEN_DRAW_CANDIDATES, CLOSE_DRAW_CANDIDATES),
        ("Away", OPEN_AWAY_CANDIDATES, CLOSE_AWAY_CANDIDATES),
    ]:
        opening, _ = coalesce_numeric_columns(data, opening_cols)
        closing, _ = coalesce_numeric_columns(data, closing_cols)
        valid = _valid_odds_mask(opening, closing)
        result_valid_frames.append(valid)
        data[f"{prefix}_OddsMove"] = (closing - opening).where(
            valid,
            0.0,
        ).fillna(0.0)
        data[f"{prefix}_ImpliedMove"] = (
            (1.0 / closing) - (1.0 / opening)
        ).where(valid, 0.0).fillna(0.0)

    if result_valid_frames:
        result_valid = pd.concat(result_valid_frames, axis=1).all(axis=1)
    else:
        result_valid = pd.Series(False, index=data.index)
    data["Result_ClosingAvailable"] = result_valid.astype(float)
    return data


def calculate_elo_expected_score(rating_a: float, rating_b: float) -> float:
    """Calcula a expectativa Elo de A contra B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def calculate_elo_feature_values(
    home_elo: float,
    away_elo: float,
    home_advantage: float,
) -> dict[str, float]:
    """Monta features Elo conhecidas antes da partida."""
    adjusted_home_elo = home_elo + home_advantage
    expected_home = calculate_elo_expected_score(adjusted_home_elo, away_elo)

    return {
        "Home_Elo_Pre": home_elo,
        "Away_Elo_Pre": away_elo,
        "Elo_Diff": home_elo - away_elo,
        "Elo_Home_Adv_Diff": adjusted_home_elo - away_elo,
        "Elo_Expected_Home": expected_home,
        "Elo_Expected_Away": 1.0 - expected_home,
    }


def calculate_match_result_score(home_goals: float, away_goals: float) -> float:
    """Converte placar em score Elo do mandante."""
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def calculate_margin_multiplier(home_goals: float, away_goals: float) -> float:
    """Aumenta levemente o ajuste Elo em vitorias por placar elastico."""
    goal_diff = abs(float(home_goals) - float(away_goals))
    if goal_diff <= 1.0:
        return 1.0
    return float(np.log1p(goal_diff))


def update_elo_ratings(
    ratings: dict[str, float],
    home_team: str,
    away_team: str,
    home_goals: float,
    away_goals: float,
    initial_rating: float,
    k_factor: float,
    home_advantage: float,
) -> None:
    """Atualiza ratings Elo depois que a partida terminou."""
    home_elo = ratings.get(home_team, initial_rating)
    away_elo = ratings.get(away_team, initial_rating)
    expected_home = calculate_elo_expected_score(
        home_elo + home_advantage,
        away_elo,
    )
    actual_home = calculate_match_result_score(home_goals, away_goals)
    margin_multiplier = calculate_margin_multiplier(home_goals, away_goals)
    elo_delta = k_factor * margin_multiplier * (actual_home - expected_home)

    ratings[home_team] = home_elo + elo_delta
    ratings[away_team] = away_elo - elo_delta


def add_elo_features(
    data: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 65.0,
) -> pd.DataFrame:
    """Adiciona ratings Elo pre-jogo e atualiza apos cada partida.

    As colunas criadas sao conhecidas antes da partida atual. O resultado
    da partida so entra no rating depois que suas features ja foram gravadas.
    """
    data = data.sort_values(
        ["MatchDatetime", "Liga", "Temporada", "HomeTeam", "AwayTeam"],
        kind="mergesort",
    ).reset_index(drop=True)
    data = data.copy()
    ratings: dict[str, float] = {}
    feature_rows: list[dict[str, float]] = []

    for _, row in data.iterrows():
        home_team = str(row["HomeTeam"])
        away_team = str(row["AwayTeam"])
        home_elo = ratings.get(home_team, initial_rating)
        away_elo = ratings.get(away_team, initial_rating)
        feature_rows.append(
            calculate_elo_feature_values(
                home_elo,
                away_elo,
                home_advantage,
            )
        )
        update_elo_ratings(
            ratings,
            home_team,
            away_team,
            row["FTHG"],
            row["FTAG"],
            initial_rating,
            k_factor,
            home_advantage,
        )

    elo_features = pd.DataFrame(feature_rows, index=data.index)
    return pd.concat([data, elo_features], axis=1)


def build_current_elo_ratings(
    data: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 65.0,
) -> dict[str, float]:
    """Calcula o rating Elo atual de cada time apos o historico."""
    ratings: dict[str, float] = {}
    history = data.sort_values(
        ["MatchDatetime", "Liga", "Temporada", "HomeTeam", "AwayTeam"],
        kind="mergesort",
    )

    for _, row in history.iterrows():
        update_elo_ratings(
            ratings,
            str(row["HomeTeam"]),
            str(row["AwayTeam"]),
            row["FTHG"],
            row["FTAG"],
            initial_rating,
            k_factor,
            home_advantage,
        )

    return ratings


def add_elo_features_to_fixtures(
    fixtures: pd.DataFrame,
    ratings: dict[str, float],
    initial_rating: float = 1500.0,
    home_advantage: float = 65.0,
) -> pd.DataFrame:
    """Adiciona features Elo atuais em jogos futuros."""
    if fixtures.empty:
        return fixtures.copy()

    fixtures = fixtures.copy()
    feature_rows = []
    for _, row in fixtures.iterrows():
        home_elo = ratings.get(str(row["HomeTeam"]), initial_rating)
        away_elo = ratings.get(str(row["AwayTeam"]), initial_rating)
        feature_rows.append(
            calculate_elo_feature_values(
                home_elo,
                away_elo,
                home_advantage,
            )
        )

    elo_features = pd.DataFrame(feature_rows, index=fixtures.index)
    return pd.concat([fixtures, elo_features], axis=1)


def _initial_standing() -> dict[str, float]:
    """Cria uma linha zerada de tabela da temporada."""
    return {
        "played": 0.0,
        "points": 0.0,
        "gf": 0.0,
        "ga": 0.0,
        "gd": 0.0,
    }


def _ranking_frame(standings: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Monta ranking da liga antes da rodada atual."""
    rows = []
    for team, stats in standings.items():
        rows.append(
            {
                "Team": team,
                "Played": stats["played"],
                "Points": stats["points"],
                "GF": stats["gf"],
                "GA": stats["ga"],
                "GD": stats["gd"],
                "PPG": (
                    stats["points"] / stats["played"]
                    if stats["played"] > 0
                    else 0.0
                ),
            }
        )

    ranking = pd.DataFrame(rows).sort_values(
        ["Points", "GD", "GF", "Team"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    ranking["Rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def _pressure_score(
    rank: int,
    points: float,
    ranking: pd.DataFrame,
    team_count: int,
    season_progress: float,
    pressure_type: str,
) -> float:
    """Calcula uma pressao contextual normalizada entre 0 e 1."""
    end_stage = float(np.clip((season_progress - 0.45) / 0.40, 0.0, 1.0))
    if end_stage == 0.0 or team_count < 4:
        return 0.0

    if pressure_type == "title":
        leader_points = float(ranking.iloc[0]["Points"])
        gap = leader_points - points
        rank_gate = rank <= min(4, team_count)
        return end_stage * max(0.0, 1.0 - (gap / 9.0)) * float(rank_gate)

    danger_rank = max(team_count - 3, 1)
    danger_points = float(
        ranking.loc[ranking["Rank"].eq(danger_rank), "Points"].iloc[0]
    )
    gap = points - danger_points
    rank_gate = rank >= max(danger_rank - 3, 1)
    return end_stage * max(0.0, 1.0 - (abs(gap) / 9.0)) * float(rank_gate)


def _standing_context_features(
    standings: dict[str, dict[str, float]],
    home_team: str,
    away_team: str,
    team_count: int,
    rounds_total: int,
) -> dict[str, float]:
    """Cria features de importancia antes da partida."""
    ranking = _ranking_frame(standings)
    home = ranking[ranking["Team"].eq(home_team)].iloc[0]
    away = ranking[ranking["Team"].eq(away_team)].iloc[0]
    avg_played = (float(home["Played"]) + float(away["Played"])) / 2.0
    season_progress = avg_played / rounds_total if rounds_total else 0.0

    home_title = _pressure_score(
        int(home["Rank"]),
        float(home["Points"]),
        ranking,
        team_count,
        season_progress,
        "title",
    )
    away_title = _pressure_score(
        int(away["Rank"]),
        float(away["Points"]),
        ranking,
        team_count,
        season_progress,
        "title",
    )
    home_relegation = _pressure_score(
        int(home["Rank"]),
        float(home["Points"]),
        ranking,
        team_count,
        season_progress,
        "relegation",
    )
    away_relegation = _pressure_score(
        int(away["Rank"]),
        float(away["Points"]),
        ranking,
        team_count,
        season_progress,
        "relegation",
    )
    home_importance = max(home_title, home_relegation)
    away_importance = max(away_title, away_relegation)

    return {
        "Home_PreMatch_Rank": float(home["Rank"]),
        "Away_PreMatch_Rank": float(away["Rank"]),
        "Rank_Diff": float(away["Rank"] - home["Rank"]),
        "Home_PreMatch_PointsPerGame": float(home["PPG"]),
        "Away_PreMatch_PointsPerGame": float(away["PPG"]),
        "PointsPerGame_Diff": float(home["PPG"] - away["PPG"]),
        "Home_TitlePressure": home_title,
        "Away_TitlePressure": away_title,
        "Home_RelegationPressure": home_relegation,
        "Away_RelegationPressure": away_relegation,
        "Home_MatchImportance": home_importance,
        "Away_MatchImportance": away_importance,
        "MatchImportance": max(home_importance, away_importance),
        "Importance_Diff": home_importance - away_importance,
        "TopClash": float(
            int(home["Rank"]) <= min(6, team_count)
            and int(away["Rank"]) <= min(6, team_count)
        )
        * float(np.clip((season_progress - 0.35) / 0.50, 0.0, 1.0)),
        "SeasonProgress": float(np.clip(season_progress, 0.0, 1.0)),
    }


def _update_standing(
    standings: dict[str, dict[str, float]],
    home_team: str,
    away_team: str,
    home_goals: float,
    away_goals: float,
) -> None:
    """Atualiza a tabela da liga apos uma partida concluida."""
    home = standings[home_team]
    away = standings[away_team]
    home["played"] += 1.0
    away["played"] += 1.0
    home["gf"] += home_goals
    home["ga"] += away_goals
    away["gf"] += away_goals
    away["ga"] += home_goals
    home["gd"] = home["gf"] - home["ga"]
    away["gd"] = away["gf"] - away["ga"]

    if home_goals > away_goals:
        home["points"] += 3.0
    elif home_goals < away_goals:
        away["points"] += 3.0
    else:
        home["points"] += 1.0
        away["points"] += 1.0


def add_match_importance_features(data: pd.DataFrame) -> pd.DataFrame:
    """Adiciona contexto de tabela antes do jogo sem vazar resultados."""
    if data.empty:
        return data

    required_cols = {
        "Liga",
        "Temporada",
        "MatchDatetime",
        "HomeTeam",
        "AwayTeam",
    }
    if not required_cols.issubset(data.columns):
        return data

    enriched = data.copy()
    for col in MATCH_IMPORTANCE_FEATURE_COLS:
        enriched[col] = np.nan

    grouped = enriched.groupby(["Liga", "Temporada"], sort=False)
    for _, group in grouped:
        group = group.sort_values(
            ["MatchDatetime", "MatchId"],
            kind="mergesort",
        )
        teams = sorted(
            set(group["HomeTeam"].dropna()).union(group["AwayTeam"].dropna())
        )
        team_count = len(teams)
        if team_count < 2:
            continue

        standings = {team: _initial_standing() for team in teams}
        rounds_total = max((team_count - 1) * 2, 1)

        for _, time_group in group.groupby("MatchDatetime", sort=False):
            for index, row in time_group.iterrows():
                home_team = row["HomeTeam"]
                away_team = row["AwayTeam"]
                if pd.isna(home_team) or pd.isna(away_team):
                    continue

                standings.setdefault(home_team, _initial_standing())
                standings.setdefault(away_team, _initial_standing())
                features = _standing_context_features(
                    standings,
                    home_team,
                    away_team,
                    team_count,
                    rounds_total,
                )
                for col, value in features.items():
                    enriched.at[index, col] = value

            for _, row in time_group.iterrows():
                if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                    continue
                _update_standing(
                    standings,
                    row["HomeTeam"],
                    row["AwayTeam"],
                    float(row["FTHG"]),
                    float(row["FTAG"]),
                )

    return enriched


def _first_existing_column(
    columns: Sequence[str],
    candidates: Sequence[str],
) -> str | None:
    """Retorna o primeiro nome de coluna disponivel em uma lista."""
    normalized = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        column = normalized.get(candidate.lower())
        if column is not None:
            return column
    return None


def _numeric_lineup_column(
    lineups: pd.DataFrame,
    candidates: Sequence[str],
    default: float,
) -> pd.Series:
    """Le uma coluna numerica de escalação com valor padrao."""
    column = _first_existing_column(lineups.columns, candidates)
    if column is None:
        return pd.Series(default, index=lineups.index, dtype="float64")
    return pd.to_numeric(lineups[column], errors="coerce").fillna(default)


def _boolean_lineup_column(
    lineups: pd.DataFrame,
    candidates: Sequence[str],
) -> pd.Series:
    """Converte flags textuais/numericas de escalacao confirmada."""
    column = _first_existing_column(lineups.columns, candidates)
    if column is None:
        return pd.Series(0.0, index=lineups.index, dtype="float64")

    raw_values = lineups[column]
    numeric_values = pd.to_numeric(raw_values, errors="coerce")
    text_values = raw_values.astype(str).str.strip().str.lower()
    mapped_values = text_values.map(
        {
            "1": 1.0,
            "true": 1.0,
            "yes": 1.0,
            "sim": 1.0,
            "confirmed": 1.0,
            "confirmada": 1.0,
            "0": 0.0,
            "false": 0.0,
            "no": 0.0,
            "nao": 0.0,
            "não": 0.0,
            "provavel": 0.0,
            "provável": 0.0,
        }
    )
    return numeric_values.fillna(mapped_values).fillna(0.0).astype(float)


def _load_lineup_features(lineup_features_path: Path) -> pd.DataFrame:
    """Carrega um CSV opcional de escalacoes/desfalques normalizado por time."""
    try:
        lineups = pd.read_csv(lineup_features_path)
    except (OSError, pd.errors.EmptyDataError) as exc:
        print(f"[lineups] Nao foi possivel ler {lineup_features_path}: {exc}")
        return pd.DataFrame()

    date_col = _first_existing_column(lineups.columns, ["MatchDate", "Date", "Data"])
    team_col = _first_existing_column(lineups.columns, ["Team", "Time", "Equipe"])
    if date_col is None or team_col is None:
        print(
            "[lineups] CSV ignorado: informe colunas Date/MatchDate e Team."
        )
        return pd.DataFrame()

    league_col = _first_existing_column(lineups.columns, ["Liga", "League", "Div"])
    normalized = pd.DataFrame(
        {
            "MatchDate": pd.to_datetime(
                lineups[date_col],
                dayfirst=True,
                errors="coerce",
            ).dt.date,
            "Team": lineups[team_col].astype(str).str.strip(),
            "LineupDataAvailable": 1.0,
            "LineupConfirmed": _boolean_lineup_column(
                lineups,
                ["LineupConfirmed", "IsConfirmed", "Confirmed", "Confirmada"],
            ),
            "LineupStrength": _numeric_lineup_column(
                lineups,
                ["LineupStrength", "Strength", "ForcaEscalacao"],
                1.0,
            ),
            "MissingStarters": _numeric_lineup_column(
                lineups,
                ["MissingStarters", "DesfalquesTitulares", "MissingXI"],
                0.0,
            ),
            "MissingKeyPlayers": _numeric_lineup_column(
                lineups,
                ["MissingKeyPlayers", "DesfalquesChave", "KeyAbsences"],
                0.0,
            ),
        }
    )
    if league_col is not None:
        normalized["Liga"] = lineups[league_col].astype(str).str.strip()

    normalized = normalized.dropna(subset=["MatchDate", "Team"])
    if normalized.empty:
        return normalized

    sort_cols = ["MatchDate", "Team", "LineupConfirmed"]
    if "Liga" in normalized.columns:
        sort_cols.insert(1, "Liga")
    return normalized.sort_values(sort_cols).drop_duplicates(
        subset=[col for col in ["MatchDate", "Liga", "Team"] if col in normalized],
        keep="last",
    )


def _match_date_series(data: pd.DataFrame) -> pd.Series:
    """Extrai a data civil da partida para combinar fontes externas."""
    if "MatchDatetime" in data.columns:
        return pd.to_datetime(data["MatchDatetime"], errors="coerce").dt.date
    return pd.to_datetime(data["Date"], dayfirst=True, errors="coerce").dt.date


def _assign_lineup_side(
    enriched: pd.DataFrame,
    lineups: pd.DataFrame,
    side: str,
) -> None:
    """Combina dados de escalacao para mandante ou visitante."""
    team_col = f"{side}Team"
    left = pd.DataFrame(
        {
            "__index": enriched.index,
            "MatchDate": _match_date_series(enriched),
            team_col: enriched[team_col].astype(str).str.strip(),
        }
    )
    left_on = ["MatchDate", team_col]
    right_on = ["MatchDate", "Team"]

    if "Liga" in enriched.columns and "Liga" in lineups.columns:
        left["Liga"] = enriched["Liga"].astype(str).str.strip()
        left_on.insert(1, "Liga")
        right_on.insert(1, "Liga")

    merged = left.merge(
        lineups,
        how="left",
        left_on=left_on,
        right_on=right_on,
        sort=False,
    ).set_index("__index")
    prefix = "Home" if side == "Home" else "Away"
    defaults = {
        "LineupDataAvailable": 0.0,
        "LineupConfirmed": 0.0,
        "LineupStrength": 1.0,
        "MissingStarters": 0.0,
        "MissingKeyPlayers": 0.0,
    }
    for col, default in defaults.items():
        enriched.loc[merged.index, f"{prefix}_{col}"] = (
            merged[col].fillna(default).astype(float)
        )


def add_lineup_features(
    data: pd.DataFrame,
    lineup_features_path: Path | None = None,
) -> pd.DataFrame:
    """Adiciona features opcionais de escalacao/desfalques.

    O CSV esperado e por time e partida, com colunas minimas `Date` e `Team`.
    Colunas opcionais aceitas: `Liga`, `LineupConfirmed`, `LineupStrength`,
    `MissingStarters` e `MissingKeyPlayers`.
    """
    if data.empty:
        return data.copy()

    required_cols = {"HomeTeam", "AwayTeam"}
    if not required_cols.issubset(data.columns):
        return data.copy()

    enriched = data.copy()
    defaults = {
        "Home_LineupDataAvailable": 0.0,
        "Away_LineupDataAvailable": 0.0,
        "Home_LineupConfirmed": 0.0,
        "Away_LineupConfirmed": 0.0,
        "Home_LineupStrength": 1.0,
        "Away_LineupStrength": 1.0,
        "LineupStrength_Diff": 0.0,
        "Home_MissingStarters": 0.0,
        "Away_MissingStarters": 0.0,
        "MissingStarters_Diff": 0.0,
        "Home_MissingKeyPlayers": 0.0,
        "Away_MissingKeyPlayers": 0.0,
        "MissingKeyPlayers_Diff": 0.0,
    }
    for col, default in defaults.items():
        enriched[col] = default

    if lineup_features_path is None:
        return enriched

    lineup_features_path = Path(lineup_features_path)
    if not lineup_features_path.exists():
        return enriched

    lineups = _load_lineup_features(lineup_features_path)
    if lineups.empty:
        return enriched

    _assign_lineup_side(enriched, lineups, "Home")
    _assign_lineup_side(enriched, lineups, "Away")
    enriched["LineupStrength_Diff"] = (
        enriched["Home_LineupStrength"] - enriched["Away_LineupStrength"]
    )
    enriched["MissingStarters_Diff"] = (
        enriched["Away_MissingStarters"] - enriched["Home_MissingStarters"]
    )
    enriched["MissingKeyPlayers_Diff"] = (
        enriched["Away_MissingKeyPlayers"] - enriched["Home_MissingKeyPlayers"]
    )
    print(f"[lineups] Features carregadas de: {lineup_features_path}")
    return enriched


def _numeric_column_or_nan(data: pd.DataFrame, column: str) -> pd.Series:
    """Retorna coluna numerica ou NaN quando ela nao existe."""
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    return pd.to_numeric(data[column], errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Calcula uma razao numerica protegida contra divisao invalida."""
    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")
    ratio = numerator / denominator.replace(0.0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def add_match_stats_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria diferencas e totais a partir das estatisticas moveis."""
    enriched = data.copy()
    for col in MATCH_STATS_FEATURE_COLS:
        if col not in enriched.columns:
            enriched[col] = 0.0

    enriched["Shots_Diff_Roll5"] = (
        enriched["Home_ShotsFor_Roll5"] - enriched["Away_ShotsFor_Roll5"]
    )
    enriched["ShotsAllowed_Diff_Roll5"] = (
        enriched["Home_ShotsAgainst_Roll5"] - enriched["Away_ShotsAgainst_Roll5"]
    )
    enriched["Shots_Total_Roll5"] = (
        enriched["Home_ShotsFor_Roll5"] + enriched["Away_ShotsFor_Roll5"]
    )
    enriched["ShotsOnTarget_Diff_Roll5"] = (
        enriched["Home_ShotsOnTargetFor_Roll5"]
        - enriched["Away_ShotsOnTargetFor_Roll5"]
    )
    enriched["ShotsOnTargetAllowed_Diff_Roll5"] = (
        enriched["Home_ShotsOnTargetAgainst_Roll5"]
        - enriched["Away_ShotsOnTargetAgainst_Roll5"]
    )
    enriched["ShotsOnTarget_Total_Roll5"] = (
        enriched["Home_ShotsOnTargetFor_Roll5"]
        + enriched["Away_ShotsOnTargetFor_Roll5"]
    )
    enriched["ShotPressure_Diff_Roll5"] = (
        enriched["Home_ShotsOnTargetFor_Roll5"]
        - enriched["Home_ShotsOnTargetAgainst_Roll5"]
        - enriched["Away_ShotsOnTargetFor_Roll5"]
        + enriched["Away_ShotsOnTargetAgainst_Roll5"]
    )
    enriched["ShotAccuracy_Diff_Roll5"] = (
        enriched["Home_ShotAccuracy_Roll5"]
        - enriched["Away_ShotAccuracy_Roll5"]
    )
    enriched["ConversionRate_Diff_Roll5"] = (
        enriched["Home_ConversionRate_Roll5"]
        - enriched["Away_ConversionRate_Roll5"]
    )
    enriched["Corners_Diff_Roll5"] = (
        enriched["Home_CornersFor_Roll5"] - enriched["Away_CornersFor_Roll5"]
    )
    enriched["Corners_Total_Roll5"] = (
        enriched["Home_CornersFor_Roll5"] + enriched["Away_CornersFor_Roll5"]
    )
    home_discipline = (
        enriched["Home_YellowCardsFor_Roll5"]
        + 2.0 * enriched["Home_RedCardsFor_Roll5"]
    )
    away_discipline = (
        enriched["Away_YellowCardsFor_Roll5"]
        + 2.0 * enriched["Away_RedCardsFor_Roll5"]
    )
    enriched["Discipline_Diff_Roll5"] = home_discipline - away_discipline
    enriched["RedCards_Diff_Roll5"] = (
        enriched["Home_RedCardsFor_Roll5"] - enriched["Away_RedCardsFor_Roll5"]
    )
    enriched["FirstHalfGoals_Total_Roll5"] = (
        enriched["Home_FirstHalfTotalGoals_Roll5"]
        + enriched["Away_FirstHalfTotalGoals_Roll5"]
    )
    enriched["SecondHalfGoals_Total_Roll5"] = (
        enriched["Home_SecondHalfTotalGoals_Roll5"]
        + enriched["Away_SecondHalfTotalGoals_Roll5"]
    )
    enriched["Venue_ShotsOnTarget_Total_Roll5"] = (
        enriched["Home_Home_ShotsOnTargetFor_Roll5"]
        + enriched["Away_Away_ShotsOnTargetFor_Roll5"]
    )
    enriched["Venue_Corners_Total_Roll5"] = (
        enriched["Home_Home_CornersFor_Roll5"]
        + enriched["Away_Away_CornersFor_Roll5"]
    )
    enriched["Venue_ShotPressure_Diff_Roll5"] = (
        enriched["Home_Home_ShotsOnTargetFor_Roll5"]
        - enriched["Home_Home_ShotsOnTargetAgainst_Roll5"]
        - enriched["Away_Away_ShotsOnTargetFor_Roll5"]
        + enriched["Away_Away_ShotsOnTargetAgainst_Roll5"]
    )

    enriched[MATCH_STATS_FEATURE_COLS] = (
        enriched[MATCH_STATS_FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    return enriched


def add_xg_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria diferencas e totais a partir das medias moveis de xG."""
    enriched = data.copy()
    for col in XG_FEATURE_COLS:
        if col not in enriched.columns:
            enriched[col] = 0.0

    enriched["xG_Attack_Diff_Roll5"] = (
        enriched["Home_xGFor_Roll5"] - enriched["Away_xGFor_Roll5"]
    )
    enriched["xG_Defense_Diff_Roll5"] = (
        enriched["Home_xGAgainst_Roll5"] - enriched["Away_xGAgainst_Roll5"]
    )
    enriched["xG_Total_Roll5"] = (
        enriched["Home_xGFor_Roll5"] + enriched["Away_xGFor_Roll5"]
    )
    enriched["xG_Expected_Total_Match_Roll5"] = (
        enriched["Home_xGFor_Roll5"]
        + enriched["Away_xGAgainst_Roll5"]
        + enriched["Away_xGFor_Roll5"]
        + enriched["Home_xGAgainst_Roll5"]
    ) / 2.0
    enriched["xG_Diff_Roll5"] = (
        enriched["Home_xGDiff_Roll5"] - enriched["Away_xGDiff_Roll5"]
    )
    enriched["xG_Finishing_Diff_Roll5"] = (
        enriched["Home_FinishingDelta_Roll5"]
        - enriched["Away_FinishingDelta_Roll5"]
    )
    enriched["xG_Defensive_Delta_Diff_Roll5"] = (
        enriched["Home_DefensiveDelta_Roll5"]
        - enriched["Away_DefensiveDelta_Roll5"]
    )
    enriched["Venue_xG_Total_Roll5"] = (
        enriched["Home_Home_xGFor_Roll5"]
        + enriched["Away_Away_xGFor_Roll5"]
    )
    enriched["Venue_xG_Diff_Roll5"] = (
        enriched["Home_Home_xGDiff_Roll5"]
        - enriched["Away_Away_xGDiff_Roll5"]
    )

    enriched[XG_FEATURE_COLS] = (
        enriched[XG_FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    return enriched


def add_referee_features(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Adiciona historico pre-jogo do arbitro quando disponivel."""
    enriched = data.copy()
    for col in REFEREE_FEATURE_COLS:
        enriched[col] = 0.0

    if "Referee" not in enriched.columns:
        return enriched

    referee = enriched["Referee"].astype(str).str.strip()
    referee = referee.mask(referee.str.lower().isin(["", "nan", "none"]))
    history = pd.DataFrame(
        {
            "Referee": referee,
            "MatchDatetime": enriched["MatchDatetime"],
            "TotalGoals": enriched["FTHG"] + enriched["FTAG"],
            "TotalCards": (
                _numeric_column_or_nan(enriched, "HY").fillna(0.0)
                + _numeric_column_or_nan(enriched, "AY").fillna(0.0)
                + 2.0 * _numeric_column_or_nan(enriched, "HR").fillna(0.0)
                + 2.0 * _numeric_column_or_nan(enriched, "AR").fillna(0.0)
            ),
            "RedCards": (
                _numeric_column_or_nan(enriched, "HR").fillna(0.0)
                + _numeric_column_or_nan(enriched, "AR").fillna(0.0)
            ),
            "Fouls": (
                _numeric_column_or_nan(enriched, "HF").fillna(0.0)
                + _numeric_column_or_nan(enriched, "AF").fillna(0.0)
            ),
            "DataAvailable": referee.notna().astype(float),
        },
        index=enriched.index,
    ).dropna(subset=["Referee", "MatchDatetime"])
    if history.empty:
        return enriched

    history = history.sort_values(
        ["Referee", "MatchDatetime"],
        kind="mergesort",
    )
    grouped = history.groupby("Referee", group_keys=False)
    rolling_specs = {
        "Referee_DataAvailable_Roll20": "DataAvailable",
        "Referee_TotalGoals_Roll20": "TotalGoals",
        "Referee_TotalCards_Roll20": "TotalCards",
        "Referee_RedCards_Roll20": "RedCards",
        "Referee_Fouls_Roll20": "Fouls",
    }
    for output_col, source_col in rolling_specs.items():
        history[output_col] = grouped[source_col].transform(
            lambda series: series.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )

    enriched.loc[history.index, REFEREE_FEATURE_COLS] = history[
        REFEREE_FEATURE_COLS
    ].fillna(0.0)
    return enriched


def add_referee_features_to_fixtures(
    historical_data: pd.DataFrame,
    fixtures: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Adiciona historico do arbitro em fixtures futuras quando conhecido."""
    enriched = fixtures.copy()
    for col in REFEREE_FEATURE_COLS:
        enriched[col] = 0.0

    if fixtures.empty or "Referee" not in fixtures.columns or "Referee" not in historical_data.columns:
        return enriched

    history = add_referee_features(historical_data, window=window)
    referee_history = history.dropna(subset=["Referee", "MatchDatetime"]).copy()
    if referee_history.empty:
        return enriched

    referee_history["Referee"] = referee_history["Referee"].astype(str).str.strip()
    for index, row in enriched.iterrows():
        referee = row.get("Referee")
        if pd.isna(referee):
            continue

        referee_matches = referee_history[
            referee_history["Referee"].eq(str(referee).strip())
            & referee_history["MatchDatetime"].lt(row["MatchDatetime"])
        ].sort_values("MatchDatetime", kind="mergesort")
        if referee_matches.empty:
            continue

        recent = referee_matches.tail(window)
        enriched.loc[index, "Referee_DataAvailable_Roll20"] = 1.0
        enriched.loc[index, "Referee_TotalGoals_Roll20"] = float(
            (recent["FTHG"] + recent["FTAG"]).mean()
        )
        enriched.loc[index, "Referee_TotalCards_Roll20"] = float(
            (
                _numeric_column_or_nan(recent, "HY").fillna(0.0)
                + _numeric_column_or_nan(recent, "AY").fillna(0.0)
                + 2.0 * _numeric_column_or_nan(recent, "HR").fillna(0.0)
                + 2.0 * _numeric_column_or_nan(recent, "AR").fillna(0.0)
            ).mean()
        )
        enriched.loc[index, "Referee_RedCards_Roll20"] = float(
            (
                _numeric_column_or_nan(recent, "HR").fillna(0.0)
                + _numeric_column_or_nan(recent, "AR").fillna(0.0)
            ).mean()
        )
        enriched.loc[index, "Referee_Fouls_Roll20"] = float(
            (
                _numeric_column_or_nan(recent, "HF").fillna(0.0)
                + _numeric_column_or_nan(recent, "AF").fillna(0.0)
            ).mean()
        )

    return enriched


def build_team_match_table(data: pd.DataFrame) -> pd.DataFrame:
    """Transforma partidas em linhas por time para calcular historico."""
    base_cols = ["MatchId", "MatchDatetime"]

    home_rows = data[
        base_cols + ["HomeTeam", "FTHG", "FTAG"]
    ].rename(
        columns={
            "HomeTeam": "Team",
            "FTHG": "GoalsFor",
            "FTAG": "GoalsAgainst",
        }
    )
    home_rows["Side"] = "home"

    away_rows = data[
        base_cols + ["AwayTeam", "FTAG", "FTHG"]
    ].rename(
        columns={
            "AwayTeam": "Team",
            "FTAG": "GoalsFor",
            "FTHG": "GoalsAgainst",
        }
    )
    away_rows["Side"] = "away"

    home_rows["ShotsFor"] = _numeric_column_or_nan(data, "HS")
    home_rows["ShotsAgainst"] = _numeric_column_or_nan(data, "AS")
    home_rows["ShotsOnTargetFor"] = _numeric_column_or_nan(data, "HST")
    home_rows["ShotsOnTargetAgainst"] = _numeric_column_or_nan(data, "AST")
    home_rows["CornersFor"] = _numeric_column_or_nan(data, "HC")
    home_rows["CornersAgainst"] = _numeric_column_or_nan(data, "AC")
    home_rows["FoulsFor"] = _numeric_column_or_nan(data, "HF")
    home_rows["FoulsAgainst"] = _numeric_column_or_nan(data, "AF")
    home_rows["YellowCardsFor"] = _numeric_column_or_nan(data, "HY")
    home_rows["YellowCardsAgainst"] = _numeric_column_or_nan(data, "AY")
    home_rows["RedCardsFor"] = _numeric_column_or_nan(data, "HR")
    home_rows["RedCardsAgainst"] = _numeric_column_or_nan(data, "AR")
    home_rows["FirstHalfGoalsFor"] = _numeric_column_or_nan(data, "HTHG")
    home_rows["FirstHalfGoalsAgainst"] = _numeric_column_or_nan(data, "HTAG")
    home_rows["SecondHalfGoalsFor"] = (
        home_rows["GoalsFor"] - home_rows["FirstHalfGoalsFor"]
    )
    home_rows["SecondHalfGoalsAgainst"] = (
        home_rows["GoalsAgainst"] - home_rows["FirstHalfGoalsAgainst"]
    )
    home_rows["xGFor"] = _numeric_column_or_nan(data, "Understat_Home_xG")
    home_rows["xGAgainst"] = _numeric_column_or_nan(data, "Understat_Away_xG")

    away_rows["ShotsFor"] = _numeric_column_or_nan(data, "AS")
    away_rows["ShotsAgainst"] = _numeric_column_or_nan(data, "HS")
    away_rows["ShotsOnTargetFor"] = _numeric_column_or_nan(data, "AST")
    away_rows["ShotsOnTargetAgainst"] = _numeric_column_or_nan(data, "HST")
    away_rows["CornersFor"] = _numeric_column_or_nan(data, "AC")
    away_rows["CornersAgainst"] = _numeric_column_or_nan(data, "HC")
    away_rows["FoulsFor"] = _numeric_column_or_nan(data, "AF")
    away_rows["FoulsAgainst"] = _numeric_column_or_nan(data, "HF")
    away_rows["YellowCardsFor"] = _numeric_column_or_nan(data, "AY")
    away_rows["YellowCardsAgainst"] = _numeric_column_or_nan(data, "HY")
    away_rows["RedCardsFor"] = _numeric_column_or_nan(data, "AR")
    away_rows["RedCardsAgainst"] = _numeric_column_or_nan(data, "HR")
    away_rows["FirstHalfGoalsFor"] = _numeric_column_or_nan(data, "HTAG")
    away_rows["FirstHalfGoalsAgainst"] = _numeric_column_or_nan(data, "HTHG")
    away_rows["SecondHalfGoalsFor"] = (
        away_rows["GoalsFor"] - away_rows["FirstHalfGoalsFor"]
    )
    away_rows["SecondHalfGoalsAgainst"] = (
        away_rows["GoalsAgainst"] - away_rows["FirstHalfGoalsAgainst"]
    )
    away_rows["xGFor"] = _numeric_column_or_nan(data, "Understat_Away_xG")
    away_rows["xGAgainst"] = _numeric_column_or_nan(data, "Understat_Home_xG")

    team_matches = pd.concat([home_rows, away_rows], ignore_index=True)
    team_matches["TotalGoals"] = (
        team_matches["GoalsFor"] + team_matches["GoalsAgainst"]
    )
    team_matches["GoalDiff"] = (
        team_matches["GoalsFor"] - team_matches["GoalsAgainst"]
    )
    team_matches["IsOver25"] = (team_matches["TotalGoals"] > 2.5).astype(int)
    team_matches["IsWin"] = (
        team_matches["GoalsFor"] > team_matches["GoalsAgainst"]
    ).astype(int)
    team_matches["IsDraw"] = (
        team_matches["GoalsFor"] == team_matches["GoalsAgainst"]
    ).astype(int)
    team_matches["IsLoss"] = (
        team_matches["GoalsFor"] < team_matches["GoalsAgainst"]
    ).astype(int)
    team_matches["Points"] = (
        team_matches["IsWin"] * 3 + team_matches["IsDraw"]
    )
    team_matches["CleanSheet"] = team_matches["GoalsAgainst"].eq(0).astype(int)
    team_matches["Scored"] = team_matches["GoalsFor"].gt(0).astype(int)
    team_matches["FailedToScore"] = team_matches["GoalsFor"].eq(0).astype(int)
    team_matches["MatchStatsAvailable"] = team_matches[
        [
            "ShotsFor",
            "ShotsAgainst",
            "ShotsOnTargetFor",
            "ShotsOnTargetAgainst",
            "CornersFor",
            "CornersAgainst",
        ]
    ].notna().any(axis=1).astype(float)
    team_matches["ShotAccuracy"] = _safe_ratio(
        team_matches["ShotsOnTargetFor"],
        team_matches["ShotsFor"],
    )
    team_matches["SOTAllowedRate"] = _safe_ratio(
        team_matches["ShotsOnTargetAgainst"],
        team_matches["ShotsAgainst"],
    )
    team_matches["ConversionRate"] = _safe_ratio(
        team_matches["GoalsFor"],
        team_matches["ShotsFor"],
    )
    team_matches["GoalsAllowedPerShot"] = _safe_ratio(
        team_matches["GoalsAgainst"],
        team_matches["ShotsAgainst"],
    )
    team_matches["FirstHalfTotalGoals"] = (
        team_matches["FirstHalfGoalsFor"] + team_matches["FirstHalfGoalsAgainst"]
    )
    team_matches["SecondHalfTotalGoals"] = (
        team_matches["SecondHalfGoalsFor"] + team_matches["SecondHalfGoalsAgainst"]
    )
    team_matches["xGAvailable"] = team_matches[
        ["xGFor", "xGAgainst"]
    ].notna().all(axis=1).astype(float)
    team_matches["xGTotal"] = team_matches["xGFor"] + team_matches["xGAgainst"]
    team_matches["xGDiff"] = team_matches["xGFor"] - team_matches["xGAgainst"]
    team_matches["FinishingDelta"] = (
        team_matches["GoalsFor"] - team_matches["xGFor"]
    )
    team_matches["DefensiveDelta"] = (
        team_matches["GoalsAgainst"] - team_matches["xGAgainst"]
    )

    team_matches = team_matches.sort_values(
        ["Team", "MatchDatetime", "MatchId"],
        kind="mergesort",
    )
    grouped = team_matches.groupby("Team", group_keys=False)
    grouped_side = team_matches.groupby(["Team", "Side"], group_keys=False)
    team_matches["RestDays"] = (
        team_matches["MatchDatetime"] - grouped["MatchDatetime"].shift(1)
    ).dt.days.clip(lower=0, upper=21)
    team_matches["SideRestDays"] = (
        team_matches["MatchDatetime"] - grouped_side["MatchDatetime"].shift(1)
    ).dt.days.clip(lower=0, upper=35)
    return team_matches


def add_rolling_features(
    data: pd.DataFrame,
    window: int = 5,
    feature_profile: str = "extended",
    lineup_features_path: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Cria medias moveis pre-jogo para mandante e visitante.

    A funcao usa shift(1) antes do rolling para garantir que a partida atual
    nunca entre no historico usado para prever ela mesma.
    """
    data = data.copy().reset_index(drop=True)
    data["MatchId"] = np.arange(len(data))
    data = add_match_importance_features(data)
    data = add_lineup_features(data, lineup_features_path)
    data = add_referee_features(data)

    team_matches = build_team_match_table(data)
    grouped = team_matches.groupby("Team", group_keys=False)
    grouped_side = team_matches.groupby(["Team", "Side"], group_keys=False)

    team_matches["GF_Roll5"] = grouped["GoalsFor"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["GA_Roll5"] = grouped["GoalsAgainst"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["TG_Roll5"] = grouped["TotalGoals"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["Over25_Roll5"] = grouped["IsOver25"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["Points_Roll5"] = grouped["Points"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["GoalDiff_Roll5"] = grouped["GoalDiff"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["WinRate_Roll5"] = grouped["IsWin"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["DrawRate_Roll5"] = grouped["IsDraw"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["LossRate_Roll5"] = grouped["IsLoss"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["CleanSheet_Roll5"] = grouped["CleanSheet"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["FailedToScore_Roll5"] = grouped["FailedToScore"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["GF_Side_Roll5"] = grouped_side["GoalsFor"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["GA_Side_Roll5"] = grouped_side["GoalsAgainst"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["TG_Side_Roll5"] = grouped_side["TotalGoals"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["Over25_Side_Roll5"] = grouped_side["IsOver25"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["Points_Side_Roll5"] = grouped_side["Points"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["WinRate_Side_Roll5"] = grouped_side["IsWin"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    team_matches["GoalDiff_Side_Roll5"] = grouped_side["GoalDiff"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).mean()
    )
    for metric in TEAM_MATCH_STAT_METRICS:
        team_matches[f"{metric}_Roll5"] = grouped[metric].transform(
            lambda series: series.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )
    for metric in VENUE_MATCH_STAT_METRICS:
        team_matches[f"{metric}_Side_Roll5"] = grouped_side[metric].transform(
            lambda series: series.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )
    team_matches = team_matches.copy()
    for metric in TEAM_XG_METRICS:
        team_matches[f"{metric}_Roll5"] = grouped[metric].transform(
            lambda series: series.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )
    for metric in VENUE_XG_METRICS:
        team_matches[f"{metric}_Side_Roll5"] = grouped_side[metric].transform(
            lambda series: series.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )

    home_features = team_matches.loc[
        team_matches["Side"].eq("home"),
        [
            "MatchId",
            "GF_Roll5",
            "GA_Roll5",
            "TG_Roll5",
            "Over25_Roll5",
            "Points_Roll5",
            "GoalDiff_Roll5",
            "WinRate_Roll5",
            "DrawRate_Roll5",
            "LossRate_Roll5",
            "CleanSheet_Roll5",
            "FailedToScore_Roll5",
            "GF_Side_Roll5",
            "GA_Side_Roll5",
            "TG_Side_Roll5",
            "Over25_Side_Roll5",
            "Points_Side_Roll5",
            "WinRate_Side_Roll5",
            "GoalDiff_Side_Roll5",
            "RestDays",
            "SideRestDays",
            *[f"{metric}_Roll5" for metric in TEAM_MATCH_STAT_METRICS],
            *[
                f"{metric}_Side_Roll5"
                for metric in VENUE_MATCH_STAT_METRICS
            ],
            *[f"{metric}_Roll5" for metric in TEAM_XG_METRICS],
            *[f"{metric}_Side_Roll5" for metric in VENUE_XG_METRICS],
        ],
    ].rename(
        columns={
            "GF_Roll5": "Home_GF_Roll5",
            "GA_Roll5": "Home_GA_Roll5",
            "TG_Roll5": "Home_TG_Roll5",
            "Over25_Roll5": "Home_Over25_Roll5",
            "Points_Roll5": "Home_Points_Roll5",
            "GoalDiff_Roll5": "Home_GoalDiff_Roll5",
            "WinRate_Roll5": "Home_WinRate_Roll5",
            "DrawRate_Roll5": "Home_DrawRate_Roll5",
            "LossRate_Roll5": "Home_LossRate_Roll5",
            "CleanSheet_Roll5": "Home_CleanSheet_Roll5",
            "FailedToScore_Roll5": "Home_FailedToScore_Roll5",
            "GF_Side_Roll5": "Home_Home_GF_Roll5",
            "GA_Side_Roll5": "Home_Home_GA_Roll5",
            "TG_Side_Roll5": "Home_Home_TG_Roll5",
            "Over25_Side_Roll5": "Home_Home_Over25_Roll5",
            "Points_Side_Roll5": "Home_Home_Points_Roll5",
            "WinRate_Side_Roll5": "Home_Home_WinRate_Roll5",
            "GoalDiff_Side_Roll5": "Home_Home_GoalDiff_Roll5",
            "RestDays": "Home_RestDays",
            "SideRestDays": "Home_Home_RestDays",
            **{
                f"{metric}_Roll5": f"Home_{metric}_Roll5"
                for metric in TEAM_MATCH_STAT_METRICS
            },
            **{
                f"{metric}_Side_Roll5": f"Home_Home_{metric}_Roll5"
                for metric in VENUE_MATCH_STAT_METRICS
            },
            **{
                f"{metric}_Roll5": f"Home_{metric}_Roll5"
                for metric in TEAM_XG_METRICS
            },
            **{
                f"{metric}_Side_Roll5": f"Home_Home_{metric}_Roll5"
                for metric in VENUE_XG_METRICS
            },
        }
    )
    away_features = team_matches.loc[
        team_matches["Side"].eq("away"),
        [
            "MatchId",
            "GF_Roll5",
            "GA_Roll5",
            "TG_Roll5",
            "Over25_Roll5",
            "Points_Roll5",
            "GoalDiff_Roll5",
            "WinRate_Roll5",
            "DrawRate_Roll5",
            "LossRate_Roll5",
            "CleanSheet_Roll5",
            "FailedToScore_Roll5",
            "GF_Side_Roll5",
            "GA_Side_Roll5",
            "TG_Side_Roll5",
            "Over25_Side_Roll5",
            "Points_Side_Roll5",
            "WinRate_Side_Roll5",
            "GoalDiff_Side_Roll5",
            "RestDays",
            "SideRestDays",
            *[f"{metric}_Roll5" for metric in TEAM_MATCH_STAT_METRICS],
            *[
                f"{metric}_Side_Roll5"
                for metric in VENUE_MATCH_STAT_METRICS
            ],
            *[f"{metric}_Roll5" for metric in TEAM_XG_METRICS],
            *[f"{metric}_Side_Roll5" for metric in VENUE_XG_METRICS],
        ],
    ].rename(
        columns={
            "GF_Roll5": "Away_GF_Roll5",
            "GA_Roll5": "Away_GA_Roll5",
            "TG_Roll5": "Away_TG_Roll5",
            "Over25_Roll5": "Away_Over25_Roll5",
            "Points_Roll5": "Away_Points_Roll5",
            "GoalDiff_Roll5": "Away_GoalDiff_Roll5",
            "WinRate_Roll5": "Away_WinRate_Roll5",
            "DrawRate_Roll5": "Away_DrawRate_Roll5",
            "LossRate_Roll5": "Away_LossRate_Roll5",
            "CleanSheet_Roll5": "Away_CleanSheet_Roll5",
            "FailedToScore_Roll5": "Away_FailedToScore_Roll5",
            "GF_Side_Roll5": "Away_Away_GF_Roll5",
            "GA_Side_Roll5": "Away_Away_GA_Roll5",
            "TG_Side_Roll5": "Away_Away_TG_Roll5",
            "Over25_Side_Roll5": "Away_Away_Over25_Roll5",
            "Points_Side_Roll5": "Away_Away_Points_Roll5",
            "WinRate_Side_Roll5": "Away_Away_WinRate_Roll5",
            "GoalDiff_Side_Roll5": "Away_Away_GoalDiff_Roll5",
            "RestDays": "Away_RestDays",
            "SideRestDays": "Away_Away_RestDays",
            **{
                f"{metric}_Roll5": f"Away_{metric}_Roll5"
                for metric in TEAM_MATCH_STAT_METRICS
            },
            **{
                f"{metric}_Side_Roll5": f"Away_Away_{metric}_Roll5"
                for metric in VENUE_MATCH_STAT_METRICS
            },
            **{
                f"{metric}_Roll5": f"Away_{metric}_Roll5"
                for metric in TEAM_XG_METRICS
            },
            **{
                f"{metric}_Side_Roll5": f"Away_Away_{metric}_Roll5"
                for metric in VENUE_XG_METRICS
            },
        }
    )

    data = data.merge(home_features, on="MatchId", how="left")
    data = data.merge(away_features, on="MatchId", how="left")
    data = data.copy()

    data["Attack_Diff_Roll5"] = data["Home_GF_Roll5"] - data["Away_GF_Roll5"]
    data["Defense_Diff_Roll5"] = data["Home_GA_Roll5"] - data["Away_GA_Roll5"]
    data["Home_Attack_vs_Away_Defense"] = (
        data["Home_GF_Roll5"] - data["Away_GA_Roll5"]
    )
    data["Away_Attack_vs_Home_Defense"] = (
        data["Away_GF_Roll5"] - data["Home_GA_Roll5"]
    )
    data["Total_Attack_Roll5"] = data["Home_GF_Roll5"] + data["Away_GF_Roll5"]
    data["Total_Defense_Roll5"] = data["Home_GA_Roll5"] + data["Away_GA_Roll5"]
    data["Total_Goals_Form_Roll5"] = data["Home_TG_Roll5"] + data["Away_TG_Roll5"]
    data["Over25_Rate_Diff_Roll5"] = (
        data["Home_Over25_Roll5"] - data["Away_Over25_Roll5"]
    )
    data["Venue_Attack_Diff_Roll5"] = (
        data["Home_Home_GF_Roll5"] - data["Away_Away_GF_Roll5"]
    )
    data["Venue_Defense_Diff_Roll5"] = (
        data["Home_Home_GA_Roll5"] - data["Away_Away_GA_Roll5"]
    )
    data["Venue_Total_Goals_Form_Roll5"] = (
        data["Home_Home_TG_Roll5"] + data["Away_Away_TG_Roll5"]
    )
    data["Venue_Over25_Rate_Diff_Roll5"] = (
        data["Home_Home_Over25_Roll5"] - data["Away_Away_Over25_Roll5"]
    )
    data["Expected_Total_Goals_Form_Roll5"] = (
        data["Home_GF_Roll5"]
        + data["Away_GA_Roll5"]
        + data["Away_GF_Roll5"]
        + data["Home_GA_Roll5"]
    ) / 2.0
    data["Points_Diff_Roll5"] = (
        data["Home_Points_Roll5"] - data["Away_Points_Roll5"]
    )
    data["GoalDiff_Diff_Roll5"] = (
        data["Home_GoalDiff_Roll5"] - data["Away_GoalDiff_Roll5"]
    )
    data["WinRate_Diff_Roll5"] = (
        data["Home_WinRate_Roll5"] - data["Away_WinRate_Roll5"]
    )
    data["LossRate_Diff_Roll5"] = (
        data["Home_LossRate_Roll5"] - data["Away_LossRate_Roll5"]
    )
    data["CleanSheet_Diff_Roll5"] = (
        data["Home_CleanSheet_Roll5"] - data["Away_CleanSheet_Roll5"]
    )
    data["FailedToScore_Diff_Roll5"] = (
        data["Home_FailedToScore_Roll5"] - data["Away_FailedToScore_Roll5"]
    )
    data["Venue_Points_Diff_Roll5"] = (
        data["Home_Home_Points_Roll5"] - data["Away_Away_Points_Roll5"]
    )
    data["Venue_WinRate_Diff_Roll5"] = (
        data["Home_Home_WinRate_Roll5"] - data["Away_Away_WinRate_Roll5"]
    )
    data["Venue_GoalDiff_Diff_Roll5"] = (
        data["Home_Home_GoalDiff_Roll5"] - data["Away_Away_GoalDiff_Roll5"]
    )
    data["RestDays_Diff"] = data["Home_RestDays"] - data["Away_RestDays"]
    data["Venue_RestDays_Diff"] = (
        data["Home_Home_RestDays"] - data["Away_Away_RestDays"]
    )
    data = add_match_stats_interaction_features(data)
    data = add_xg_interaction_features(data)

    if feature_profile not in {"base", "extended"}:
        raise ValueError("feature_profile deve ser 'base' ou 'extended'.")

    base_feature_cols = [
        "Home_GF_Roll5",
        "Home_GA_Roll5",
        "Home_TG_Roll5",
        "Home_Over25_Roll5",
        "Home_Home_GF_Roll5",
        "Home_Home_GA_Roll5",
        "Home_Home_TG_Roll5",
        "Home_Home_Over25_Roll5",
        "Away_GF_Roll5",
        "Away_GA_Roll5",
        "Away_TG_Roll5",
        "Away_Over25_Roll5",
        "Away_Away_GF_Roll5",
        "Away_Away_GA_Roll5",
        "Away_Away_TG_Roll5",
        "Away_Away_Over25_Roll5",
        "Attack_Diff_Roll5",
        "Defense_Diff_Roll5",
        "Home_Attack_vs_Away_Defense",
        "Away_Attack_vs_Home_Defense",
        "Total_Attack_Roll5",
        "Total_Defense_Roll5",
        "Total_Goals_Form_Roll5",
        "Over25_Rate_Diff_Roll5",
        "Venue_Attack_Diff_Roll5",
        "Venue_Defense_Diff_Roll5",
        "Venue_Total_Goals_Form_Roll5",
        "Venue_Over25_Rate_Diff_Roll5",
        "Expected_Total_Goals_Form_Roll5",
    ]
    extended_feature_cols = [
        "Home_Points_Roll5",
        "Home_GoalDiff_Roll5",
        "Home_WinRate_Roll5",
        "Home_DrawRate_Roll5",
        "Home_LossRate_Roll5",
        "Home_CleanSheet_Roll5",
        "Home_FailedToScore_Roll5",
        "Home_Home_Points_Roll5",
        "Home_Home_WinRate_Roll5",
        "Home_Home_GoalDiff_Roll5",
        "Home_RestDays",
        "Home_Home_RestDays",
        "Away_Points_Roll5",
        "Away_GoalDiff_Roll5",
        "Away_WinRate_Roll5",
        "Away_DrawRate_Roll5",
        "Away_LossRate_Roll5",
        "Away_CleanSheet_Roll5",
        "Away_FailedToScore_Roll5",
        "Away_Away_Points_Roll5",
        "Away_Away_WinRate_Roll5",
        "Away_Away_GoalDiff_Roll5",
        "Away_RestDays",
        "Away_Away_RestDays",
        "Points_Diff_Roll5",
        "GoalDiff_Diff_Roll5",
        "WinRate_Diff_Roll5",
        "LossRate_Diff_Roll5",
        "CleanSheet_Diff_Roll5",
        "FailedToScore_Diff_Roll5",
        "Venue_Points_Diff_Roll5",
        "Venue_WinRate_Diff_Roll5",
        "Venue_GoalDiff_Diff_Roll5",
        "RestDays_Diff",
        "Venue_RestDays_Diff",
    ]
    extended_feature_cols.extend(MATCH_IMPORTANCE_FEATURE_COLS)
    extended_feature_cols.extend(LINEUP_FEATURE_COLS)
    extended_feature_cols.extend(MATCH_STATS_FEATURE_COLS)
    extended_feature_cols.extend(REFEREE_FEATURE_COLS)
    extended_feature_cols.extend(ODDS_MOVEMENT_FEATURE_COLS)
    extended_feature_cols.extend(XG_FEATURE_COLS)

    feature_cols = base_feature_cols.copy()
    if feature_profile == "extended":
        feature_cols.extend(extended_feature_cols)
    feature_cols.extend([col for col in ELO_FEATURE_COLS if col in data.columns])

    return data, feature_cols


def add_target_and_drop_na(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Cria o alvo Over 2.5 e remove linhas sem historico suficiente."""
    data = data.copy()
    data[TARGET_COL] = ((data["FTHG"] + data["FTAG"]) > 2.5).astype(int)
    data = data[data["FTR"].isin(RESULT_LABELS)].copy()
    data[RESULT_TARGET_COL] = data["FTR"].map(
        {label: index for index, label in enumerate(RESULT_LABELS)}
    )

    rows_before = len(data)
    data = data.dropna(
        subset=list(feature_cols) + [TARGET_COL, RESULT_TARGET_COL]
    )
    data = add_no_vig_market_probabilities(data)
    data = data.sort_values("MatchDatetime", kind="mergesort").reset_index(drop=True)

    print(
        "[features] Linhas removidas por historico insuficiente/odds invalidas: "
        f"{rows_before - len(data):,}"
    )
    print(f"[features] Dataset final modelavel: {len(data):,} jogos")

    return data


def chronological_train_test_split(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    train_size: float,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide treino/teste preservando a ordem temporal."""
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size deve estar entre 0 e 1.")

    split_index = int(len(data) * train_size)
    if split_index == 0 or split_index == len(data):
        raise ValueError("Dados insuficientes para criar treino e teste.")

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    x_train = train_data.loc[:, feature_cols]
    x_test = test_data.loc[:, feature_cols]
    y_train = train_data[target_col]
    y_test = test_data[target_col]

    print(
        "[split] Treino: "
        f"{train_data['MatchDatetime'].min().date()} a "
        f"{train_data['MatchDatetime'].max().date()} "
        f"({len(train_data):,} jogos)"
    )
    print(
        "[split] Teste:  "
        f"{test_data['MatchDatetime'].min().date()} a "
        f"{test_data['MatchDatetime'].max().date()} "
        f"({len(test_data):,} jogos)"
    )

    return x_train, x_test, y_train, y_test
