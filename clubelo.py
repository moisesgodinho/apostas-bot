"""Features de forca real dos times com ClubElo e fallback interno."""

from __future__ import annotations

import re
import time
import unicodedata
from io import StringIO
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests


CLUBELO_BASE_URL = "http://api.clubelo.com/{club}"
CLUBELO_SUPPORTED_LEAGUES = [
    "E0",
    "E1",
    "SC0",
    "D1",
    "D2",
    "SP1",
    "SP2",
    "I1",
    "I2",
    "F1",
    "F2",
    "N1",
    "P1",
    "B1",
    "G1",
    "T1",
]
TEAM_STRENGTH_FEATURE_COLS = [
    "Home_ClubElo_Available",
    "Away_ClubElo_Available",
    "ClubElo_DataAvailable",
    "Home_TeamStrength",
    "Away_TeamStrength",
    "TeamStrength_Diff",
    "TeamStrength_Home_Adv_Diff",
    "TeamStrength_Expected_Home",
    "TeamStrength_Expected_Away",
]
CLUBELO_EMPTY_CACHE_TEXT = "From,To,Elo\n"

CLUBELO_ALIASES = {
    "ac milan": "Milan",
    "ath bilbao": "Athletic",
    "ath madrid": "Atletico",
    "atletico madrid": "Atletico",
    "bayern munich": "Bayern",
    "betis": "Betis",
    "bologna": "Bologna",
    "brighton": "Brighton",
    "club brugge": "ClubBrugge",
    "cologne": "Koeln",
    "dortmund": "Dortmund",
    "ein frankfurt": "Frankfurt",
    "fc porto": "Porto",
    "fiorentina": "Fiorentina",
    "freiburg": "Freiburg",
    "genoa": "Genoa",
    "hertha": "Hertha",
    "inter": "Inter",
    "juventus": "Juventus",
    "leverkusen": "Leverkusen",
    "man city": "ManCity",
    "man united": "ManUnited",
    "monchengladbach": "Gladbach",
    "monza": "Monza",
    "nott m forest": "Forest",
    "nottm forest": "Forest",
    "paris sg": "ParisSG",
    "psg": "ParisSG",
    "rb leipzig": "RBLeipzig",
    "real madrid": "RealMadrid",
    "real sociedad": "Sociedad",
    "sociedad": "Sociedad",
    "sparta rotterdam": "SpartaRotterdam",
    "spurs": "Tottenham",
    "st etienne": "StEtienne",
    "st. gilloise": "UnionSG",
    "st gilloise": "UnionSG",
    "union berlin": "UnionBerlin",
    "werder bremen": "Werder",
    "west brom": "WestBrom",
    "wolves": "Wolves",
}


def _normalize_key(value: object) -> str:
    """Normaliza texto para busca tolerante em aliases."""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_clubelo_slug(team_name: object) -> str:
    """Converte o nome do Football-Data para o identificador do ClubElo."""
    key = _normalize_key(team_name)
    if key in CLUBELO_ALIASES:
        return CLUBELO_ALIASES[key]

    original = str(team_name).strip()
    ascii_name = unicodedata.normalize("NFKD", original).encode(
        "ascii",
        "ignore",
    ).decode("ascii")
    return re.sub(r"[^A-Za-z0-9]", "", ascii_name)


def _safe_cache_name(slug: str) -> str:
    """Gera nome de arquivo seguro para o cache."""
    safe_slug = re.sub(r"[^A-Za-z0-9_-]", "_", slug).strip("_")
    return safe_slug or "unknown"


def _read_clubelo_csv(path: Path) -> pd.DataFrame:
    """Le um CSV ClubElo do cache."""
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()
    return _normalize_clubelo_history(data)


def _normalize_clubelo_history(data: pd.DataFrame) -> pd.DataFrame:
    """Padroniza tipos de uma serie historica do ClubElo."""
    required_cols = {"Elo", "From", "To"}
    if data.empty or not required_cols.issubset(data.columns):
        return pd.DataFrame()

    history = data.copy()
    history["Elo"] = pd.to_numeric(history["Elo"], errors="coerce")
    history["From"] = pd.to_datetime(history["From"], errors="coerce")
    history["To"] = pd.to_datetime(history["To"], errors="coerce")
    history = history.dropna(subset=["Elo", "From"])
    if history.empty:
        return pd.DataFrame()

    history["To"] = history["To"].fillna(pd.Timestamp.max.normalize())
    return history.sort_values("From", kind="mergesort").reset_index(drop=True)


def load_clubelo_history(
    team_name: object,
    cache_dir: Path,
    force_refresh: bool = False,
    timeout: int = 20,
) -> pd.DataFrame:
    """Baixa ou le do cache o historico ClubElo de um time."""
    slug = resolve_clubelo_slug(team_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_safe_cache_name(slug)}.csv"

    if cache_path.exists() and cache_path.stat().st_size > 0 and not force_refresh:
        return _read_clubelo_csv(cache_path)

    url = CLUBELO_BASE_URL.format(club=slug)
    try:
        response = requests.get(url, timeout=(5, timeout))
        response.raise_for_status()
    except requests.RequestException as exc:
        cache_path.write_text(CLUBELO_EMPTY_CACHE_TEXT, encoding="utf-8")
        print(
            "[clubelo] Sem resposta para "
            f"{team_name} ({slug}). Usando Elo interno. Motivo: {exc}"
        )
        return pd.DataFrame()

    cache_path.write_text(response.text, encoding="utf-8")
    time.sleep(0.20)
    try:
        return _normalize_clubelo_history(pd.read_csv(StringIO(response.text)))
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        cache_path.write_text(CLUBELO_EMPTY_CACHE_TEXT, encoding="utf-8")
        return pd.DataFrame()


def _lookup_history_by_date(
    history: pd.DataFrame,
    match_dates: pd.Series,
) -> pd.Series:
    """Busca o rating valido na data de cada partida."""
    if history.empty or match_dates.empty:
        return pd.Series(np.nan, index=match_dates.index, dtype="float64")

    left = pd.DataFrame(
        {
            "_Index": match_dates.index,
            "MatchDate": pd.to_datetime(match_dates, errors="coerce").dt.normalize(),
        }
    ).dropna(subset=["MatchDate"])
    if left.empty:
        return pd.Series(np.nan, index=match_dates.index, dtype="float64")

    merged = pd.merge_asof(
        left.sort_values("MatchDate", kind="mergesort"),
        history[["From", "To", "Elo"]].sort_values("From", kind="mergesort"),
        left_on="MatchDate",
        right_on="From",
        direction="backward",
    )
    valid = merged["To"].ge(merged["MatchDate"])
    values = pd.Series(np.nan, index=match_dates.index, dtype="float64")
    values.loc[merged.loc[valid, "_Index"]] = merged.loc[valid, "Elo"].to_numpy()
    return values


def _assign_clubelo_side(
    data: pd.DataFrame,
    cache_dir: Path,
    team_col: str,
    output_col: str,
    eligible_mask: pd.Series,
    force_refresh: bool,
) -> pd.Series:
    """Preenche ClubElo para um lado da partida."""
    values = pd.Series(np.nan, index=data.index, dtype="float64")
    teams = sorted(data.loc[eligible_mask, team_col].dropna().astype(str).unique())

    for team in teams:
        team_mask = eligible_mask & data[team_col].astype(str).eq(team)
        history = load_clubelo_history(
            team,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
        if history.empty:
            continue
        values.loc[team_mask] = _lookup_history_by_date(
            history,
            data.loc[team_mask, "MatchDatetime"],
        )

    return values.rename(output_col)


def _expected_score(rating_a: pd.Series, rating_b: pd.Series) -> pd.Series:
    """Calcula expectativa Elo vetorizada."""
    return 1.0 / (1.0 + np.power(10.0, (rating_b - rating_a) / 400.0))


def _numeric_series_or_default(
    data: pd.DataFrame,
    column: str,
    default: float,
) -> pd.Series:
    """Retorna uma serie numerica mesmo quando a coluna nao existe."""
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype="float64")
    return pd.to_numeric(data[column], errors="coerce").fillna(default)


def add_team_strength_features(
    data: pd.DataFrame,
    cache_dir: Path,
    enabled: bool = False,
    force_refresh: bool = False,
    supported_leagues: Sequence[str] | None = None,
    home_advantage: float = 65.0,
    fallback_rating: float = 1500.0,
) -> pd.DataFrame:
    """Adiciona forca dos times usando ClubElo quando disponivel.

    Quando ClubElo esta desativado ou ausente para algum clube, as features
    usam o Elo interno ja calculado no projeto. Isso evita perda de linhas no
    treino e deixa o modelo saber, via flags, quando a fonte externa existe.
    """
    enriched = data.copy()
    supported = set(supported_leagues or CLUBELO_SUPPORTED_LEAGUES)
    eligible_mask = pd.Series(False, index=enriched.index)
    if "Liga" in enriched.columns and "MatchDatetime" in enriched.columns:
        eligible_mask = enriched["Liga"].astype(str).isin(supported)

    enriched["Home_ClubElo_Pre"] = np.nan
    enriched["Away_ClubElo_Pre"] = np.nan

    if enabled and eligible_mask.any():
        enriched["Home_ClubElo_Pre"] = _assign_clubelo_side(
            enriched,
            cache_dir,
            "HomeTeam",
            "Home_ClubElo_Pre",
            eligible_mask,
            force_refresh,
        )
        enriched["Away_ClubElo_Pre"] = _assign_clubelo_side(
            enriched,
            cache_dir,
            "AwayTeam",
            "Away_ClubElo_Pre",
            eligible_mask,
            force_refresh,
        )

    home_internal = _numeric_series_or_default(
        enriched,
        "Home_Elo_Pre",
        fallback_rating,
    )
    away_internal = _numeric_series_or_default(
        enriched,
        "Away_Elo_Pre",
        fallback_rating,
    )
    home_clubelo = pd.to_numeric(enriched["Home_ClubElo_Pre"], errors="coerce")
    away_clubelo = pd.to_numeric(enriched["Away_ClubElo_Pre"], errors="coerce")

    enriched["Home_ClubElo_Available"] = home_clubelo.notna().astype(float)
    enriched["Away_ClubElo_Available"] = away_clubelo.notna().astype(float)
    enriched["ClubElo_DataAvailable"] = (
        home_clubelo.notna() & away_clubelo.notna()
    ).astype(float)

    enriched["Home_TeamStrength"] = home_clubelo.fillna(home_internal)
    enriched["Away_TeamStrength"] = away_clubelo.fillna(away_internal)
    enriched["TeamStrength_Diff"] = (
        enriched["Home_TeamStrength"] - enriched["Away_TeamStrength"]
    )
    enriched["TeamStrength_Home_Adv_Diff"] = (
        enriched["Home_TeamStrength"] + home_advantage - enriched["Away_TeamStrength"]
    )
    enriched["TeamStrength_Expected_Home"] = _expected_score(
        enriched["Home_TeamStrength"] + home_advantage,
        enriched["Away_TeamStrength"],
    )
    enriched["TeamStrength_Expected_Away"] = 1.0 - enriched[
        "TeamStrength_Expected_Home"
    ]

    clubelo_home_for_calc = home_clubelo.fillna(home_internal)
    clubelo_away_for_calc = away_clubelo.fillna(away_internal)
    enriched["ClubElo_Diff"] = clubelo_home_for_calc - clubelo_away_for_calc
    enriched["ClubElo_Home_Adv_Diff"] = (
        clubelo_home_for_calc + home_advantage - clubelo_away_for_calc
    )
    enriched["ClubElo_Expected_Home"] = _expected_score(
        clubelo_home_for_calc + home_advantage,
        clubelo_away_for_calc,
    )
    enriched["ClubElo_Expected_Away"] = 1.0 - enriched["ClubElo_Expected_Home"]

    enriched[TEAM_STRENGTH_FEATURE_COLS] = (
        enriched[TEAM_STRENGTH_FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    external_rows = int(enriched["ClubElo_DataAvailable"].sum())
    total_rows = len(enriched)
    if not enabled:
        print(
            "[forca] ClubElo desativado nesta execucao. "
            f"Usando Elo interno em {total_rows:,} jogos."
        )
    elif not eligible_mask.any():
        leagues_text = ", ".join(sorted(supported)) if supported else "nenhuma"
        print(
            "[forca] ClubElo ativado, mas este recorte nao tem ligas "
            "suportadas por essa fonte. "
            f"Ligas suportadas hoje: {leagues_text}."
        )
    else:
        print(
            "[forca] ClubElo ativado. Jogos com dados nos dois times: "
            f"{external_rows:,}/{total_rows:,}."
        )
    return enriched
