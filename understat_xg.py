"""Coleta e combinacao de xG historico do Understat."""

from __future__ import annotations

import json
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests


UNDERSTAT_BASE_URL = "https://understat.com/getLeagueData/{league}/{season}"
UNDERSTAT_LEAGUE_MAP = {
    "E0": "EPL",
    "SP1": "La_liga",
    "D1": "Bundesliga",
    "I1": "Serie_A",
    "F1": "Ligue_1",
}
TEAM_ALIASES = {
    "acmilan": "milan",
    "alaves": "deportivoalaves",
    "athbilbao": "athleticclub",
    "athmadrid": "atleticomadrid",
    "betis": "realbetis",
    "bournemouth": "afcbournemouth",
    "brighton": "brightonandhovealbion",
    "celta": "celtavigo",
    "dortmund": "borussiadortmund",
    "einfrankfurt": "eintrachtfrankfurt",
    "espanol": "espanyol",
    "fcandorra": "andorra",
    "fckoln": "fcologne",
    "herthaberlin": "herthabsc",
    "leverkusen": "bayerleverkusen",
    "manchesterutd": "manchesterunited",
    "mancity": "manchestercity",
    "manunited": "manchesterunited",
    "mgladbach": "borussiamgladbach",
    "monchengladbach": "borussiamgladbach",
    "newcastle": "newcastleunited",
    "nottmforest": "nottinghamforest",
    "parissg": "parissaintgermain",
    "psg": "parissaintgermain",
    "rb leipzig": "rbleipzig",
    "sociedad": "realsociedad",
    "stetienne": "saintetienne",
    "tottenham": "tottenhamhotspur",
    "vallecano": "rayovallecano",
    "verona": "hellasverona",
    "westbrom": "westbromwichalbion",
    "westham": "westhamunited",
    "wolves": "wolverhamptonwanderers",
}
UNDERSTAT_XG_COLUMNS = [
    "Understat_MatchId",
    "Understat_Home_xG",
    "Understat_Away_xG",
    "Understat_Total_xG",
    "Understat_Home_Win_xG_Prob",
    "Understat_Draw_xG_Prob",
    "Understat_Away_Win_xG_Prob",
    "Understat_XG_Available",
]


def normalize_team_name(value: object) -> str:
    """Normaliza nomes de times para combinar bases diferentes."""
    text = "" if pd.isna(value) else str(value).lower()
    normalized = "".join(ch for ch in text if ch.isalnum())
    return TEAM_ALIASES.get(normalized, normalized)


def football_data_season_to_understat(season: object) -> int | None:
    """Converte temporada football-data, como 2425, para ano Understat."""
    season_text = str(season)
    if len(season_text) == 4 and season_text.isdigit():
        start_year = int(season_text[:2])
        return 2000 + start_year if start_year < 70 else 1900 + start_year
    return None


def _understat_cache_path(
    cache_dir: Path,
    league_code: str,
    understat_season: int,
) -> Path:
    """Monta caminho de cache para uma liga/temporada Understat."""
    understat_league = UNDERSTAT_LEAGUE_MAP[league_code]
    return cache_dir / f"{understat_season}_{understat_league}.csv"


def _safe_float(value: object) -> float:
    """Converte valor numerico de forma tolerante."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _fetch_understat_matches(
    league_code: str,
    understat_season: int,
    timeout: int = 30,
) -> pd.DataFrame:
    """Baixa partidas de uma liga/temporada no Understat."""
    understat_league = UNDERSTAT_LEAGUE_MAP[league_code]
    url = UNDERSTAT_BASE_URL.format(
        league=understat_league,
        season=understat_season,
    )
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://understat.com/league/{understat_league}/{understat_season}",
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    response.raise_for_status()
    payload = response.json()
    matches = payload.get("dates", [])
    rows = []
    for match in matches:
        home_team = match.get("h", {}).get("title", "")
        away_team = match.get("a", {}).get("title", "")
        forecast = match.get("forecast") or {}
        rows.append(
            {
                "Liga": league_code,
                "Temporada": f"{str(understat_season)[-2:]}{str(understat_season + 1)[-2:]}",
                "UnderstatSeason": understat_season,
                "UnderstatLeague": understat_league,
                "Understat_MatchId": match.get("id"),
                "MatchDatetime": pd.to_datetime(
                    match.get("datetime"),
                    errors="coerce",
                ),
                "MatchDate": pd.to_datetime(
                    match.get("datetime"),
                    errors="coerce",
                ).date(),
                "UnderstatHomeTeam": home_team,
                "UnderstatAwayTeam": away_team,
                "HomeTeamKey": normalize_team_name(home_team),
                "AwayTeamKey": normalize_team_name(away_team),
                "Understat_Home_xG": _safe_float(
                    match.get("xG", {}).get("h")
                ),
                "Understat_Away_xG": _safe_float(
                    match.get("xG", {}).get("a")
                ),
                "Understat_Home_Win_xG_Prob": _safe_float(
                    forecast.get("w")
                ),
                "Understat_Draw_xG_Prob": _safe_float(forecast.get("d")),
                "Understat_Away_Win_xG_Prob": _safe_float(
                    forecast.get("l")
                ),
                "Understat_IsResult": bool(match.get("isResult")),
            }
        )

    data = pd.DataFrame(rows)
    if data.empty:
        return data

    data["Understat_Total_xG"] = (
        data["Understat_Home_xG"] + data["Understat_Away_xG"]
    )
    data["Understat_XG_Available"] = data[
        ["Understat_Home_xG", "Understat_Away_xG"]
    ].notna().all(axis=1).astype(float)
    return data


def load_understat_matches(
    league_code: str,
    understat_season: int,
    cache_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Carrega xG Understat com cache local por liga/temporada."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _understat_cache_path(cache_dir, league_code, understat_season)
    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        return pd.read_csv(cache_path, parse_dates=["MatchDatetime"])

    print(
        "[understat] Baixando xG "
        f"{UNDERSTAT_LEAGUE_MAP[league_code]} {understat_season}"
    )
    data = _fetch_understat_matches(league_code, understat_season)
    data.to_csv(cache_path, index=False, encoding="utf-8-sig")
    time.sleep(1)
    return data


def _load_supported_understat_data(
    data: pd.DataFrame,
    cache_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Carrega todos os recortes Understat necessarios ao dataset."""
    frames = []
    supported = data[data["Liga"].isin(UNDERSTAT_LEAGUE_MAP)].copy()
    if supported.empty:
        return pd.DataFrame()

    seasons = (
        supported[["Liga", "Temporada"]]
        .drop_duplicates()
        .sort_values(["Liga", "Temporada"], kind="mergesort")
    )
    for _, row in seasons.iterrows():
        understat_season = football_data_season_to_understat(row["Temporada"])
        league_code = str(row["Liga"])
        if understat_season is None or league_code not in UNDERSTAT_LEAGUE_MAP:
            continue

        try:
            frame = load_understat_matches(
                league_code,
                understat_season,
                cache_dir,
                force_refresh=force_refresh,
            )
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            print(
                "[understat] Falha ao carregar "
                f"{league_code} {understat_season}: {exc}"
            )
            continue

        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["Liga"] = combined["Liga"].astype(str)
    combined["Temporada"] = combined["Temporada"].astype(str)
    combined["MatchDate"] = pd.to_datetime(
        combined["MatchDate"],
        errors="coerce",
    ).dt.date
    return combined


def _similarity(left: str, right: str) -> float:
    """Calcula similaridade simples entre dois nomes normalizados."""
    return SequenceMatcher(None, left, right).ratio()


def _apply_fuzzy_xg_matches(
    merged: pd.DataFrame,
    xg_data: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve partidas nao combinadas por nome exato usando fuzzy matching."""
    unresolved = merged[
        merged["Understat_XG_Available"].isna()
        & merged["Liga"].isin(UNDERSTAT_LEAGUE_MAP)
    ]
    if unresolved.empty or xg_data.empty:
        return merged

    xg_groups = {
        key: group
        for key, group in xg_data.groupby(["Liga", "Temporada", "MatchDate"])
    }
    for index, row in unresolved.iterrows():
        key = (row["Liga"], row["Temporada"], row["MatchDate"])
        candidates = xg_groups.get(key)
        if candidates is None or candidates.empty:
            continue

        home_key = row["HomeTeamKey"]
        away_key = row["AwayTeamKey"]
        best_score = 0.0
        best_row = None
        for _, candidate in candidates.iterrows():
            score = (
                _similarity(home_key, candidate["HomeTeamKey"])
                + _similarity(away_key, candidate["AwayTeamKey"])
            ) / 2.0
            if score > best_score:
                best_score = score
                best_row = candidate

        if best_row is None or best_score < 0.82:
            continue

        for col in UNDERSTAT_XG_COLUMNS:
            merged.at[index, col] = best_row[col]

    return merged


def merge_understat_xg_features(
    data: pd.DataFrame,
    cache_dir: Path,
    enabled: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Adiciona xG Understat em partidas historicas quando disponivel."""
    enriched = data.copy()
    for col in UNDERSTAT_XG_COLUMNS:
        if col not in enriched.columns:
            enriched[col] = np.nan

    if not enabled or enriched.empty:
        enriched["Understat_XG_Available"] = enriched[
            "Understat_XG_Available"
        ].fillna(0.0)
        return enriched

    required_cols = {"Liga", "Temporada", "MatchDatetime", "HomeTeam", "AwayTeam"}
    if not required_cols.issubset(enriched.columns):
        enriched["Understat_XG_Available"] = enriched[
            "Understat_XG_Available"
        ].fillna(0.0)
        return enriched

    xg_data = _load_supported_understat_data(
        enriched,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    if xg_data.empty:
        enriched["Understat_XG_Available"] = enriched[
            "Understat_XG_Available"
        ].fillna(0.0)
        return enriched

    left = (
        enriched.drop(columns=UNDERSTAT_XG_COLUMNS, errors="ignore")
        .reset_index(names="__row_id")
        .copy()
    )
    left["MatchDate"] = pd.to_datetime(left["MatchDatetime"]).dt.date
    left["Liga"] = left["Liga"].astype(str)
    left["Temporada"] = left["Temporada"].astype(str)
    left["HomeTeamKey"] = left["HomeTeam"].map(normalize_team_name)
    left["AwayTeamKey"] = left["AwayTeam"].map(normalize_team_name)

    right_cols = [
        "Liga",
        "Temporada",
        "MatchDate",
        "HomeTeamKey",
        "AwayTeamKey",
        *UNDERSTAT_XG_COLUMNS,
    ]
    merged = left.merge(
        xg_data[right_cols],
        how="left",
        on=["Liga", "Temporada", "MatchDate", "HomeTeamKey", "AwayTeamKey"],
        sort=False,
    )
    merged = _apply_fuzzy_xg_matches(merged, xg_data)
    merged = merged.sort_values("__row_id", kind="mergesort")

    for col in UNDERSTAT_XG_COLUMNS:
        enriched[col] = merged[col].to_numpy()
    enriched["Understat_XG_Available"] = enriched[
        "Understat_XG_Available"
    ].fillna(0.0)

    matched = int(enriched["Understat_XG_Available"].sum())
    supported_rows = int(enriched["Liga"].isin(UNDERSTAT_LEAGUE_MAP).sum())
    print(
        "[understat] xG combinado: "
        f"{matched:,}/{supported_rows:,} jogos das ligas suportadas"
    )
    return enriched
