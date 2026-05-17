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

    team_matches = pd.concat([home_rows, away_rows], ignore_index=True)
    team_matches["TotalGoals"] = (
        team_matches["GoalsFor"] + team_matches["GoalsAgainst"]
    )
    team_matches["IsOver25"] = (team_matches["TotalGoals"] > 2.5).astype(int)
    return team_matches.sort_values(
        ["Team", "MatchDatetime", "MatchId"],
        kind="mergesort",
    )


def add_rolling_features(
    data: pd.DataFrame,
    window: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """Cria medias moveis pre-jogo para mandante e visitante.

    A funcao usa shift(1) antes do rolling para garantir que a partida atual
    nunca entre no historico usado para prever ela mesma.
    """
    data = data.copy().reset_index(drop=True)
    data["MatchId"] = np.arange(len(data))

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

    home_features = team_matches.loc[
        team_matches["Side"].eq("home"),
        [
            "MatchId",
            "GF_Roll5",
            "GA_Roll5",
            "TG_Roll5",
            "Over25_Roll5",
            "GF_Side_Roll5",
            "GA_Side_Roll5",
            "TG_Side_Roll5",
            "Over25_Side_Roll5",
        ],
    ].rename(
        columns={
            "GF_Roll5": "Home_GF_Roll5",
            "GA_Roll5": "Home_GA_Roll5",
            "TG_Roll5": "Home_TG_Roll5",
            "Over25_Roll5": "Home_Over25_Roll5",
            "GF_Side_Roll5": "Home_Home_GF_Roll5",
            "GA_Side_Roll5": "Home_Home_GA_Roll5",
            "TG_Side_Roll5": "Home_Home_TG_Roll5",
            "Over25_Side_Roll5": "Home_Home_Over25_Roll5",
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
            "GF_Side_Roll5",
            "GA_Side_Roll5",
            "TG_Side_Roll5",
            "Over25_Side_Roll5",
        ],
    ].rename(
        columns={
            "GF_Roll5": "Away_GF_Roll5",
            "GA_Roll5": "Away_GA_Roll5",
            "TG_Roll5": "Away_TG_Roll5",
            "Over25_Roll5": "Away_Over25_Roll5",
            "GF_Side_Roll5": "Away_Away_GF_Roll5",
            "GA_Side_Roll5": "Away_Away_GA_Roll5",
            "TG_Side_Roll5": "Away_Away_TG_Roll5",
            "Over25_Side_Roll5": "Away_Away_Over25_Roll5",
        }
    )

    data = data.merge(home_features, on="MatchId", how="left")
    data = data.merge(away_features, on="MatchId", how="left")

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

    feature_cols = [
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
