"""Coleta e normalizacao de jogos futuros e odds por casa."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from config import (
    FIXTURES_URL,
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
    RAW_IMPLIED_AWAY_COL,
    RAW_IMPLIED_DRAW_COL,
    RAW_IMPLIED_HOME_COL,
    RAW_IMPLIED_OVER_COL,
    RAW_IMPLIED_UNDER_COL,
)
from data_pipeline import (
    AWAY_ODDS_CANDIDATES,
    DRAW_ODDS_CANDIDATES,
    HOME_ODDS_CANDIDATES,
    OVER_CANDIDATES,
    UNDER_CANDIDATES,
    add_no_vig_market_probabilities,
    add_market_quality_features,
    add_odds_movement_features,
    coalesce_numeric_columns,
    parse_match_datetime,
)


BOOKMAKER_1X2_COLUMNS = {
    "Bet365": {
        "Casa": ["B365CH", "B365H"],
        "Empate": ["B365CD", "B365D"],
        "Fora": ["B365CA", "B365A"],
    },
    "Betfair Sportsbook": {
        "Casa": ["BFDCH", "BFDH"],
        "Empate": ["BFDCD", "BFDD"],
        "Fora": ["BFDCA", "BFDA"],
    },
    "Betfair Exchange": {
        "Casa": ["BFECH", "BFEH"],
        "Empate": ["BFECD", "BFED"],
        "Fora": ["BFECA", "BFEA"],
    },
    "BetMGM": {
        "Casa": ["BMGMCH", "BMGMH"],
        "Empate": ["BMGMCD", "BMGMD"],
        "Fora": ["BMGMCA", "BMGMA"],
    },
    "BetVictor": {
        "Casa": ["BVCH", "BVH"],
        "Empate": ["BVCD", "BVD"],
        "Fora": ["BVCA", "BVA"],
    },
    "Betway": {
        "Casa": ["BWCH", "BWH"],
        "Empate": ["BWCD", "BWD"],
        "Fora": ["BWCA", "BWA"],
    },
    "Coral": {
        "Casa": ["CLCH", "CLH"],
        "Empate": ["CLCD", "CLD"],
        "Fora": ["CLCA", "CLA"],
    },
    "Ladbrokes": {
        "Casa": ["LBCH", "LBH"],
        "Empate": ["LBCD", "LBD"],
        "Fora": ["LBCA", "LBA"],
    },
    "Pinnacle": {
        "Casa": ["PSCH", "PSH"],
        "Empate": ["PSCD", "PSD"],
        "Fora": ["PSCA", "PSA"],
    },
}

BOOKMAKER_TOTALS_COLUMNS = {
    "Bet365": {
        "Over 2.5": ["B365C>2.5", "B365>2.5"],
        "Under 2.5": ["B365C<2.5", "B365<2.5"],
    },
    "Betfair Exchange": {
        "Over 2.5": ["BFEC>2.5", "BFE>2.5"],
        "Under 2.5": ["BFEC<2.5", "BFE<2.5"],
    },
    "Pinnacle": {
        "Over 2.5": ["PC>2.5", "P>2.5"],
        "Under 2.5": ["PC<2.5", "P<2.5"],
    },
}
BOOKMAKER_NAME_ALIASES = {
    "bet365": "Bet365",
    "betfair exchange": "Betfair Exchange",
    "betfair sportsbook": "Betfair Sportsbook",
    "betmgm": "BetMGM",
    "betvictor": "BetVictor",
    "betway": "Betway",
    "betano": "Betano",
    "coral": "Coral",
    "ladbrokes": "Ladbrokes",
    "pinnacle": "Pinnacle",
}

BEST_SELECTION_COLUMNS = {
    "Casa": ("Best_Odd_H", "Best_Bookmaker_H"),
    "Empate": ("Best_Odd_D", "Best_Bookmaker_D"),
    "Fora": ("Best_Odd_A", "Best_Bookmaker_A"),
    "Over 2.5": ("Best_Odd_Over25", "Best_Bookmaker_Over25"),
    "Under 2.5": ("Best_Odd_Under25", "Best_Bookmaker_Under25"),
}
SELECTED_SELECTION_COLUMNS = {
    "Casa": ("Selected_Odd_H", "Selected_Bookmaker_H"),
    "Empate": ("Selected_Odd_D", "Selected_Bookmaker_D"),
    "Fora": ("Selected_Odd_A", "Selected_Bookmaker_A"),
    "Over 2.5": ("Selected_Odd_Over25", "Selected_Bookmaker_Over25"),
    "Under 2.5": ("Selected_Odd_Under25", "Selected_Bookmaker_Under25"),
}
LEAGUE_TIMEZONE_MAP = {
    "E0": "Europe/London",
    "E1": "Europe/London",
    "SC0": "Europe/London",
    "D1": "Europe/Berlin",
    "D2": "Europe/Berlin",
    "SP1": "Europe/Madrid",
    "SP2": "Europe/Madrid",
    "I1": "Europe/Rome",
    "I2": "Europe/Rome",
    "F1": "Europe/Paris",
    "F2": "Europe/Paris",
    "N1": "Europe/Amsterdam",
    "P1": "Europe/Lisbon",
    "B1": "Europe/Brussels",
    "G1": "Europe/Athens",
    "T1": "Europe/Istanbul",
    "BRA": "America/Sao_Paulo",
}
BRAZIL_TZ = ZoneInfo("America/Sao_Paulo")


def canonical_bookmaker_name(bookmaker: str | None) -> str | None:
    """Normaliza o nome da casa para uma chave canonica."""
    if bookmaker is None:
        return None
    normalized = str(bookmaker).strip().lower()
    return BOOKMAKER_NAME_ALIASES.get(normalized)


def bookmaker_is_supported(bookmaker: str | None) -> bool:
    """Indica se a casa existe no mapeamento atual da fonte."""
    canonical = canonical_bookmaker_name(bookmaker)
    if canonical is None:
        return False
    return canonical in BOOKMAKER_1X2_COLUMNS or canonical in BOOKMAKER_TOTALS_COLUMNS


def add_selected_bookmaker_odds(
    fixtures: pd.DataFrame,
    preferred_bookmaker: str | None = None,
    allow_fallback_to_best: bool = True,
) -> pd.DataFrame:
    """Resolve as odds que serao usadas nos palpites futuros.

    Quando uma casa preferida e informada, as colunas `Selected_*` usam apenas
    as odds dessa casa. Sem preferencia, elas apontam para a melhor odd
    disponivel no dataset. Quando o fallback esta liberado, o sistema tenta a
    casa preferida primeiro e cai para a melhor odd disponivel se ela nao
    aparecer na fonte atual.
    """
    enriched = fixtures.copy()
    canonical = canonical_bookmaker_name(preferred_bookmaker)
    supported = bookmaker_is_supported(preferred_bookmaker)
    requested = canonical or (
        str(preferred_bookmaker).strip() if preferred_bookmaker else "Melhor disponivel"
    )

    enriched["RequestedBookmaker"] = requested
    enriched["UsesPreferredBookmaker"] = bool(preferred_bookmaker)
    enriched["RequestedBookmakerSupported"] = float(supported)
    enriched["AllowBookmakerFallback"] = bool(allow_fallback_to_best)

    if not preferred_bookmaker:
        for selection, (selected_odd_col, selected_bookmaker_col) in (
            SELECTED_SELECTION_COLUMNS.items()
        ):
            best_odd_col, best_bookmaker_col = BEST_SELECTION_COLUMNS[selection]
            enriched[selected_odd_col] = pd.to_numeric(
                enriched[best_odd_col],
                errors="coerce",
            )
            enriched[selected_bookmaker_col] = enriched[best_bookmaker_col].fillna("")
        return enriched

    one_x_two_columns = BOOKMAKER_1X2_COLUMNS.get(canonical, {})
    totals_columns = BOOKMAKER_TOTALS_COLUMNS.get(canonical, {})

    for selection in ["Casa", "Empate", "Fora"]:
        selected_odd_col, selected_bookmaker_col = SELECTED_SELECTION_COLUMNS[selection]
        best_odd_col, best_bookmaker_col = BEST_SELECTION_COLUMNS[selection]
        columns = one_x_two_columns.get(selection, [])
        preferred_series = enriched.apply(
            lambda row: _first_valid_odd(row, columns),
            axis=1,
        )
        preferred_valid = pd.to_numeric(preferred_series, errors="coerce").notna()
        if allow_fallback_to_best:
            best_series = pd.to_numeric(enriched[best_odd_col], errors="coerce")
            enriched[selected_odd_col] = np.where(
                preferred_valid,
                preferred_series,
                best_series,
            )
            enriched[selected_bookmaker_col] = np.where(
                preferred_valid,
                requested,
                enriched[best_bookmaker_col].fillna(""),
            )
        else:
            enriched[selected_odd_col] = preferred_series
            enriched[selected_bookmaker_col] = np.where(
                preferred_valid,
                requested,
                "",
            )

    for selection in ["Over 2.5", "Under 2.5"]:
        selected_odd_col, selected_bookmaker_col = SELECTED_SELECTION_COLUMNS[selection]
        best_odd_col, best_bookmaker_col = BEST_SELECTION_COLUMNS[selection]
        columns = totals_columns.get(selection, [])
        preferred_series = enriched.apply(
            lambda row: _first_valid_odd(row, columns),
            axis=1,
        )
        preferred_valid = pd.to_numeric(preferred_series, errors="coerce").notna()
        if allow_fallback_to_best:
            best_series = pd.to_numeric(enriched[best_odd_col], errors="coerce")
            enriched[selected_odd_col] = np.where(
                preferred_valid,
                preferred_series,
                best_series,
            )
            enriched[selected_bookmaker_col] = np.where(
                preferred_valid,
                requested,
                enriched[best_bookmaker_col].fillna(""),
            )
        else:
            enriched[selected_odd_col] = preferred_series
            enriched[selected_bookmaker_col] = np.where(
                preferred_valid,
                requested,
                "",
            )

    return enriched


def download_fixtures_if_needed(
    raw_dir: Path,
    force_refresh: bool = False,
    max_cache_age_hours: int = 6,
    timeout: int = 30,
) -> Path:
    """Baixa o CSV publico de fixtures se o cache local estiver vencido."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / "fixtures.csv"

    if file_path.exists() and file_path.stat().st_size > 0 and not force_refresh:
        modified_at = datetime.fromtimestamp(file_path.stat().st_mtime)
        cache_age = datetime.now() - modified_at
        if cache_age <= timedelta(hours=max_cache_age_hours):
            print(f"[cache] Usando fixtures locais: {file_path}")
            return file_path

    print(f"[download] Baixando fixtures: {FIXTURES_URL}")
    response = requests.get(FIXTURES_URL, timeout=timeout)
    response.raise_for_status()
    file_path.write_bytes(response.content)
    time.sleep(1)
    return file_path


def _valid_decimal_odd(value: object) -> float:
    """Converte uma odd decimal em float ou retorna NaN."""
    try:
        odd = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(odd) or odd <= 1.0:
        return np.nan
    return odd


def _first_valid_odd(row: pd.Series, columns: Sequence[str]) -> float:
    """Retorna a primeira odd valida conforme a prioridade de colunas."""
    for column in columns:
        if column not in row.index:
            continue
        odd = _valid_decimal_odd(row[column])
        if np.isfinite(odd):
            return odd
    return np.nan


def _best_bookmaker_odd(
    row: pd.Series,
    bookmaker_columns: dict[str, dict[str, list[str]]],
    selection: str,
) -> tuple[float, str]:
    """Encontra a maior odd disponivel para uma selecao."""
    best_odd = np.nan
    best_bookmaker = ""

    for bookmaker, selection_columns in bookmaker_columns.items():
        odd = _first_valid_odd(row, selection_columns.get(selection, []))
        if np.isfinite(odd) and (not np.isfinite(best_odd) or odd > best_odd):
            best_odd = odd
            best_bookmaker = bookmaker

    return best_odd, best_bookmaker


def add_best_bookmaker_odds(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Adiciona melhores odds e respectivas casas para cada selecao."""
    fixtures = fixtures.copy()

    for selection in ["Casa", "Empate", "Fora"]:
        odd_col, bookmaker_col = BEST_SELECTION_COLUMNS[selection]
        best_values = fixtures.apply(
            lambda row: _best_bookmaker_odd(
                row,
                BOOKMAKER_1X2_COLUMNS,
                selection,
            ),
            axis=1,
            result_type="expand",
        )
        fixtures[odd_col] = pd.to_numeric(best_values[0], errors="coerce")
        fixtures[bookmaker_col] = best_values[1].fillna("")

    for selection in ["Over 2.5", "Under 2.5"]:
        odd_col, bookmaker_col = BEST_SELECTION_COLUMNS[selection]
        best_values = fixtures.apply(
            lambda row: _best_bookmaker_odd(
                row,
                BOOKMAKER_TOTALS_COLUMNS,
                selection,
            ),
            axis=1,
            result_type="expand",
        )
        fixtures[odd_col] = pd.to_numeric(best_values[0], errors="coerce")
        fixtures[bookmaker_col] = best_values[1].fillna("")

    return fixtures


def build_bookmaker_odds_long(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Transforma odds por casa em formato longo para visualizacao."""
    rows: list[dict[str, object]] = []

    for _, row in fixtures.iterrows():
        base_row = {
            "FixtureId": row["FixtureId"],
            "MatchDatetime": row["MatchDatetime"],
            "MatchDatetimeBR": row.get("MatchDatetimeBR", pd.NaT),
            "KickoffTimezone": row.get("KickoffTimezone", ""),
            "Liga": row["Liga"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
        }

        for bookmaker, selections in BOOKMAKER_1X2_COLUMNS.items():
            for selection, columns in selections.items():
                odd = _first_valid_odd(row, columns)
                if np.isfinite(odd):
                    rows.append(
                        {
                            **base_row,
                            "Market": "Resultado 1X2",
                            "Selection": selection,
                            "Bookmaker": bookmaker,
                            "Odd": odd,
                        }
                    )

        for bookmaker, selections in BOOKMAKER_TOTALS_COLUMNS.items():
            for selection, columns in selections.items():
                odd = _first_valid_odd(row, columns)
                if np.isfinite(odd):
                    rows.append(
                        {
                            **base_row,
                            "Market": "Total de Gols",
                            "Selection": selection,
                            "Bookmaker": bookmaker,
                            "Odd": odd,
                        }
                    )

    return pd.DataFrame(rows)


def _fixture_datetime_to_utc(row: pd.Series) -> pd.Timestamp:
    """Converte o horario local da liga para UTC."""
    match_datetime = row["MatchDatetime"]
    if pd.isna(match_datetime):
        return pd.NaT

    timezone_name = LEAGUE_TIMEZONE_MAP.get(str(row["Liga"]), "UTC")
    local_datetime = match_datetime.to_pydatetime().replace(
        tzinfo=ZoneInfo(timezone_name)
    )
    return pd.Timestamp(local_datetime.astimezone(timezone.utc))


def add_fixture_timezone_columns(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas UTC e horario de Brasilia para fixtures."""
    fixtures = fixtures.copy()
    fixtures["KickoffTimezone"] = fixtures["Liga"].map(
        lambda league: LEAGUE_TIMEZONE_MAP.get(str(league), "UTC")
    )
    fixtures["MatchDatetimeUTC"] = pd.to_datetime(
        fixtures.apply(_fixture_datetime_to_utc, axis=1),
        utc=True,
    )
    fixtures["MatchDatetimeBR"] = (
        fixtures["MatchDatetimeUTC"]
        .dt.tz_convert(BRAZIL_TZ)
        .dt.tz_localize(None)
    )
    return fixtures


def load_upcoming_fixtures(
    raw_dir: Path,
    leagues: Sequence[str],
    days_ahead: int = 7,
    force_refresh: bool = False,
    preferred_bookmaker: str | None = None,
    allow_bookmaker_fallback: bool = True,
) -> pd.DataFrame:
    """Carrega fixtures futuras do Football-Data com odds normalizadas."""
    file_path = download_fixtures_if_needed(
        raw_dir,
        force_refresh=force_refresh,
    )
    fixtures = pd.read_csv(file_path, encoding="utf-8-sig")
    fixtures.columns = fixtures.columns.str.strip()

    required_cols = ["Div", "Date", "HomeTeam", "AwayTeam"]
    missing_cols = [col for col in required_cols if col not in fixtures.columns]
    if missing_cols:
        raise KeyError(f"Colunas obrigatorias ausentes em fixtures: {missing_cols}")

    fixtures = fixtures.rename(columns={"Div": "Liga"}).copy()
    fixtures["Temporada"] = "upcoming"
    fixtures = parse_match_datetime(fixtures)
    fixtures = fixtures.dropna(
        subset=["MatchDatetime", "Liga", "HomeTeam", "AwayTeam"]
    ).copy()
    fixtures = add_fixture_timezone_columns(fixtures)

    now_utc = pd.Timestamp.now(tz=timezone.utc).floor("min")
    end_date_utc = now_utc + pd.Timedelta(days=days_ahead)
    fixtures = fixtures[
        fixtures["MatchDatetimeUTC"].between(
            now_utc,
            end_date_utc,
            inclusive="both",
        )
        & fixtures["Liga"].isin(leagues)
    ].copy()

    if fixtures.empty:
        return fixtures

    fixtures = fixtures.sort_values(
        ["MatchDatetime", "Liga", "HomeTeam", "AwayTeam"],
        kind="mergesort",
    ).reset_index(drop=True)
    fixtures["FixtureId"] = (
        fixtures["Liga"].astype(str)
        + "|"
        + fixtures["MatchDatetime"].dt.strftime("%Y-%m-%d %H:%M")
        + "|"
        + fixtures["HomeTeam"].astype(str)
        + "|"
        + fixtures["AwayTeam"].astype(str)
    )

    fixtures[ODD_OVER_COL], _ = coalesce_numeric_columns(
        fixtures,
        OVER_CANDIDATES,
    )
    fixtures[ODD_UNDER_COL], _ = coalesce_numeric_columns(
        fixtures,
        UNDER_CANDIDATES,
    )
    fixtures[ODD_HOME_COL], _ = coalesce_numeric_columns(
        fixtures,
        HOME_ODDS_CANDIDATES,
    )
    fixtures[ODD_DRAW_COL], _ = coalesce_numeric_columns(
        fixtures,
        DRAW_ODDS_CANDIDATES,
    )
    fixtures[ODD_AWAY_COL], _ = coalesce_numeric_columns(
        fixtures,
        AWAY_ODDS_CANDIDATES,
    )
    fixtures = add_no_vig_market_probabilities(fixtures)
    fixtures = add_odds_movement_features(fixtures)
    fixtures = add_market_quality_features(fixtures)
    fixtures = add_best_bookmaker_odds(fixtures)
    fixtures = add_selected_bookmaker_odds(
        fixtures,
        preferred_bookmaker=preferred_bookmaker,
        allow_fallback_to_best=allow_bookmaker_fallback,
    )

    print(
        "[fixtures] Jogos futuros carregados: "
        f"{len(fixtures):,} ({now_utc} a {end_date_utc})"
    )
    return fixtures


def implied_probability_from_best_odd(data: pd.DataFrame, odd_col: str) -> pd.Series:
    """Calcula probabilidade implicita bruta a partir da melhor odd."""
    return 1.0 / pd.to_numeric(data[odd_col], errors="coerce")


def has_over_under_odds(data: pd.DataFrame) -> pd.Series:
    """Indica linhas com odds suficientes para o mercado Over/Under."""
    return (
        data[
            [
                ODD_OVER_COL,
                ODD_UNDER_COL,
                RAW_IMPLIED_OVER_COL,
                RAW_IMPLIED_UNDER_COL,
                NO_VIG_OVER_COL,
                NO_VIG_UNDER_COL,
            ]
        ]
        .notna()
        .all(axis=1)
    )


def has_result_odds(data: pd.DataFrame) -> pd.Series:
    """Indica linhas com odds suficientes para o mercado 1X2."""
    required_cols = [
        ODD_HOME_COL,
        ODD_DRAW_COL,
        ODD_AWAY_COL,
        RAW_IMPLIED_HOME_COL,
        RAW_IMPLIED_DRAW_COL,
        RAW_IMPLIED_AWAY_COL,
        NO_VIG_HOME_COL,
        NO_VIG_DRAW_COL,
        NO_VIG_AWAY_COL,
    ]
    return data[required_cols].notna().all(axis=1)
