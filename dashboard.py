"""Dashboard local para analisar backtests de apostas +EV.

Execute com:
    streamlit run dashboard.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DEFAULT_LEAGUES, DEFAULT_SEASONS


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OVER25_PATH = OUTPUT_DIR / "backtest_over25_results.csv"
UNDER25_PATH = OUTPUT_DIR / "backtest_under25_results.csv"
RESULT_1X2_PATH = OUTPUT_DIR / "backtest_result_1x2_results.csv"
WIN_PATH = OUTPUT_DIR / "backtest_win_results.csv"
UPCOMING_PATH = OUTPUT_DIR / "upcoming_predictions.csv"
UPCOMING_ODDS_PATH = OUTPUT_DIR / "upcoming_odds_by_bookmaker.csv"
UPCOMING_CONTEXT_PATH = OUTPUT_DIR / "upcoming_context_summary.csv"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison_summary.csv"
FILTER_OPTIMIZATION_PATH = OUTPUT_DIR / "filter_optimization_summary.csv"
FILTER_OPTIMIZATION_GRID_PATH = OUTPUT_DIR / "filter_optimization_grid.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance_summary.csv"
XGB_TUNING_PATH = OUTPUT_DIR / "xgb_tuning_summary.csv"
PROBABILITY_BLEND_PATH = OUTPUT_DIR / "probability_blend_summary.csv"
CALIBRATION_CURVE_PATH = OUTPUT_DIR / "calibration_curve_summary.csv"
CALIBRATION_METRICS_PATH = OUTPUT_DIR / "calibration_metrics_summary.csv"
REALISTIC_SUMMARY_PATH = OUTPUT_DIR / "realistic_backtest_summary.csv"
REALISTIC_GRID_PATH = OUTPUT_DIR / "realistic_backtest_grid.csv"
REALISTIC_BETS_PATH = OUTPUT_DIR / "realistic_backtest_bets.csv"
REALISTIC_MONTHLY_PATH = OUTPUT_DIR / "realistic_backtest_monthly.csv"
REALISTIC_LEAGUE_PATH = OUTPUT_DIR / "realistic_backtest_league.csv"
BACKTEST_MARKETS = [
    "Over 2.5",
    "Under 2.5",
    "Resultado 1X2",
    "Vitoria Casa/Fora",
]
BACKTEST_MARKET_PATHS = {
    "Over 2.5": OVER25_PATH,
    "Under 2.5": UNDER25_PATH,
    "Resultado 1X2": RESULT_1X2_PATH,
    "Vitoria Casa/Fora": WIN_PATH,
}
PIPELINE_PATH = BASE_DIR / "over25_ev_model.py"
PREDICT_UPCOMING_PATH = BASE_DIR / "predict_upcoming.py"
LEAGUE_NAME_MAP = {
    "E0": "Inglaterra - Premier League",
    "E1": "Inglaterra - Championship",
    "SC0": "Escocia - Premiership",
    "D1": "Alemanha - Bundesliga",
    "D2": "Alemanha - 2. Bundesliga",
    "SP1": "Espanha - La Liga",
    "SP2": "Espanha - La Liga 2",
    "I1": "Italia - Serie A",
    "I2": "Italia - Serie B",
    "F1": "Franca - Ligue 1",
    "F2": "Franca - Ligue 2",
    "N1": "Holanda - Eredivisie",
    "P1": "Portugal - Primeira Liga",
    "B1": "Belgica - Pro League",
    "G1": "Grecia - Super League",
    "T1": "Turquia - Super Lig",
    "BRA": "Brasil - Serie A",
}
LEAGUE_PRESET_MAP = {
    "Todas": DEFAULT_LEAGUES,
    "Top 5 Europa": ["E0", "D1", "SP1", "I1", "F1"],
    "2as divisoes": ["E1", "D2", "SP2", "I2", "F2"],
    "Sem Brasil": [league for league in DEFAULT_LEAGUES if league != "BRA"],
    "Brasil": ["BRA"],
}


def league_label(code: str) -> str:
    """Retorna o nome amigavel da liga."""
    return LEAGUE_NAME_MAP.get(code, code)


def league_competition_label(code: str) -> str:
    """Retorna apenas o nome do campeonato, sem o pais."""
    label = league_label(code)
    if " - " not in label:
        return label
    return label.split(" - ", maxsplit=1)[1]


def league_country(code: str) -> str:
    """Extrai o pais ou regiao do nome da liga."""
    return league_label(code).split(" - ", maxsplit=1)[0]


def ordered_leagues(leagues: list[str]) -> list[str]:
    """Ordena ligas pelo nome amigavel."""
    return sorted(leagues, key=lambda code: league_label(code))


def valid_codes(codes: list[str], options: list[str]) -> list[str]:
    """Mantem apenas codigos existentes nas opcoes disponiveis."""
    option_set = set(options)
    return [code for code in codes if code in option_set]


def build_league_presets(
    options: list[str],
    default: list[str],
) -> dict[str, list[str]]:
    """Cria presets de ligas compativeis com as opcoes visiveis."""
    presets = {
        name: valid_codes(codes, options)
        for name, codes in LEAGUE_PRESET_MAP.items()
    }
    presets = {name: codes for name, codes in presets.items() if codes}
    if not presets:
        presets["Disponiveis"] = valid_codes(default, options) or options
    return presets


def selected_league_caption(selected: list[str], visible_count: int) -> str:
    """Resume campeonatos selecionados sem poluir a sidebar."""
    if not selected:
        return "Nenhum campeonato selecionado."

    names = [league_competition_label(code) for code in selected[:3]]
    suffix = "" if len(selected) <= 3 else f" +{len(selected) - 3}"
    return f"{len(selected)} de {visible_count}: {', '.join(names)}{suffix}"


def render_league_selector(
    label: str,
    options: list[str],
    default: list[str],
    key: str,
    *,
    sidebar: bool = False,
) -> list[str]:
    """Renderiza seletor visual de ligas com pais, preset e ajuste fino."""
    available = ordered_leagues(options)
    default_available = valid_codes(default, available) or available
    container_fn = st.sidebar.container if sidebar else st.container

    with container_fn(border=True):
        st.markdown(f"**{label}**")
        countries = sorted({league_country(code) for code in available})
        selected_countries = st.pills(
            "Pais",
            options=countries,
            selection_mode="multi",
            default=[],
            help="Deixe vazio para mostrar todos os paises.",
            key=f"{key}_country_pills_v2",
            width="stretch",
        )
        selected_countries = selected_countries or []
        if selected_countries:
            visible = [
                code
                for code in available
                if league_country(code) in set(selected_countries)
            ]
        else:
            visible = available

        country_key = "_".join(selected_countries or ["Todos"]).replace(" ", "_")
        presets = build_league_presets(visible, default_available)
        preset_names = list(presets)
        selected_preset = st.segmented_control(
            "Pacote",
            options=preset_names,
            default=preset_names[0],
            key=f"{key}_preset_{country_key}",
            width="stretch",
        )
        selected_preset = selected_preset or preset_names[0]
        preset_default = presets[selected_preset]
        compact_labels = bool(
            selected_countries
            and len(selected_countries) == 1
        )
        selected = st.pills(
            "Campeonatos",
            options=visible,
            selection_mode="multi",
            default=preset_default,
            format_func=(
                league_competition_label if compact_labels else league_label
            ),
            key=f"{key}_league_pills_{country_key}_{selected_preset}",
            width="stretch",
        )
        selected = selected or []
        if not selected:
            st.warning("Selecione ao menos um campeonato.")
        st.caption(selected_league_caption(selected, len(visible)))
        return selected


st.set_page_config(
    page_title="Apostasbot | Backtests +EV",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_csv(path: Path, modified_at: float) -> pd.DataFrame:
    """Carrega um CSV de backtest com datas normalizadas."""
    _ = modified_at
    data = pd.read_csv(path, low_memory=False)
    data["MatchDatetime"] = pd.to_datetime(data["MatchDatetime"], errors="coerce")
    data = data.dropna(subset=["MatchDatetime"]).copy()
    data["MatchDate"] = data["MatchDatetime"].dt.date
    data["LigaNome"] = data["Liga"].map(league_label)
    return data


@st.cache_data(show_spinner=False)
def load_optional_csv(path: Path, modified_at: float) -> pd.DataFrame:
    """Carrega CSV opcional, tratando arquivos vazios."""
    _ = modified_at
    try:
        data = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    if "MatchDatetime" in data.columns:
        data["MatchDatetime"] = pd.to_datetime(
            data["MatchDatetime"],
            errors="coerce",
        )
        data = data.dropna(subset=["MatchDatetime"]).copy()
        data["MatchDate"] = data["MatchDatetime"].dt.date
    if "MatchDatetimeBR" in data.columns:
        data["MatchDatetimeBR"] = pd.to_datetime(
            data["MatchDatetimeBR"],
            errors="coerce",
        )
    elif "MatchDatetime" in data.columns:
        data["MatchDatetimeBR"] = data["MatchDatetime"]
    if "Liga" in data.columns:
        data["LigaNome"] = data["Liga"].map(league_label)
    return data


def format_money(value: float) -> str:
    """Formata valor monetario em reais."""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct(value: float) -> str:
    """Formata percentual."""
    return f"{value:.2%}".replace(".", ",")


def format_table_pct(value: object) -> str:
    """Formata proporcoes como percentual para tabelas estilizadas."""
    if pd.isna(value):
        return ""
    try:
        return format_pct(float(value))
    except (TypeError, ValueError):
        return str(value)


def format_table_money(value: object) -> str:
    """Formata dinheiro para tabelas estilizadas."""
    if pd.isna(value):
        return ""
    try:
        return format_money(float(value))
    except (TypeError, ValueError):
        return str(value)


def format_table_decimal(value: object) -> str:
    """Formata decimais simples para tabelas estilizadas."""
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.2f}".replace(".", ",")
    except (TypeError, ValueError):
        return str(value)


STATUS_STYLE_COLUMNS = {
    "Escalacao",
    "Importancia",
    "Leitura",
    "Status",
    "StatusAposta",
    "Decisao",
}
SIGNED_STYLE_COLUMNS = {
    "AvgEdge",
    "CalibrationBias",
    "CalibrationGap",
    "ChanceMedia",
    "ChanceMercado",
    "DeltaAccuracy",
    "DeltaBets",
    "DeltaProfit",
    "DeltaROI",
    "Edge",
    "EdgeMedio",
    "EvalROI",
    "EvalTotalProfit",
    "ImprovementVsMarket",
    "ImprovementVsModel",
    "Lucro",
    "LineupStrength_Diff",
    "LucroPorAposta",
    "MarketEdge",
    "MissingKeyPlayers_Diff",
    "MissingStarters_Diff",
    "Nota",
    "ROI",
    "Retorno",
    "RoiGap",
    "RuleEvalROI",
    "SimProfit",
    "SobraAcerto",
    "TestROI",
    "TestTotalProfit",
    "TotalProfit",
    "TuneROI",
    "TuneTotalProfit",
    "ValROI",
    "ValTotalProfit",
    "VantagemMedia",
    "ValueGap",
    "xG_Diff_Roll5",
}
INVERTED_SIGNED_STYLE_COLUMNS = {
    "DeltaBrierScore",
    "DeltaCalibrationECE",
    "DeltaLogLoss",
}
PERCENT_STYLE_COLUMNS = {
    "Accuracy",
    "Acerto",
    "AcertoNecessario",
    "AvgEdge",
    "CalibrationBias",
    "CalibrationGap",
    "ChanceMedia",
    "ChanceMercado",
    "DeltaAccuracy",
    "DeltaCalibrationECE",
    "DeltaROI",
    "Edge",
    "EdgeMedio",
    "EvalHitRate",
    "EvalROI",
    "HitRate",
    "ImpliedProb",
    "MarketEdge",
    "MinModelProb",
    "ModelProb",
    "NoVigProb",
    "PositiveROIRate",
    "ROI",
    "Retorno",
    "RoiGap",
    "RuleEvalROI",
    "SobraAcerto",
    "TestHitRate",
    "TestROI",
    "TuneHitRate",
    "TuneROI",
    "ValROI",
    "VantagemMedia",
    "ValueGap",
}
MONEY_STYLE_COLUMNS = {
    "Apostado",
    "EvalTotalProfit",
    "Lucro",
    "LucroPorAposta",
    "SimProfit",
    "TestMaxDrawdown",
    "TestTotalProfit",
    "TotalProfit",
    "TuneTotalProfit",
    "ValTotalProfit",
    "total_profit",
}
DECIMAL_STYLE_COLUMNS = {
    "AvgOdd",
    "BestOdd",
    "CotacaoMedia",
    "EvalAvgOdd",
    "OddMedia",
    "SelectionOdd",
    "TestAvgOdd",
    "avg_odd",
}
STATUS_COLOR_MAP = {
    "+EV": ("#dcfce7", "#166534"),
    "Alta": ("", "#b91c1c"),
    "Baixa": ("", "#64748b"),
    "Bloqueada": ("#fee2e2", "#991b1b"),
    "Confirmada": ("", "#166534"),
    "Green": ("#dcfce7", "#166534"),
    "Liberada": ("#dcfce7", "#166534"),
    "Media": ("", "#b45309"),
    "Overfit": ("#fef3c7", "#92400e"),
    "Passar": ("#f1f5f9", "#475569"),
    "Pouco volume": ("#e2e8f0", "#334155"),
    "Provavel": ("", "#2563eb"),
    "Poucos jogos": ("#e2e8f0", "#334155"),
    "Promissor": ("", "#2563eb"),
    "Forte": ("#dcfce7", "#166534"),
    "Evitar": ("#f1f5f9", "#475569"),
    "Instavel": ("", "#b91c1c"),
    "Red": ("#fee2e2", "#991b1b"),
    "Sem +EV": ("#f1f5f9", "#475569"),
}


def signed_number_style(value: object, *, inverted: bool = False) -> str:
    """Aplica cor em numeros positivos e negativos."""
    if pd.isna(value):
        return ""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return ""

    if numeric_value == 0:
        return "color: #64748b;"

    is_positive = numeric_value > 0
    if inverted:
        is_positive = not is_positive

    color = "#15803d" if is_positive else "#b91c1c"
    return f"color: {color}; font-weight: 700;"


def status_cell_style(value: object) -> str:
    """Aplica cor no texto para status operacionais."""
    background, color = STATUS_COLOR_MAP.get(str(value), ("", ""))
    _ = background
    if not color:
        return ""
    return f"color: {color}; font-weight: 700;"


def style_dashboard_table(data: pd.DataFrame):
    """Estiliza tabelas do dashboard com cores para status e sinais."""
    styler = data.style
    formatters = {}
    formatters.update(
        {
            column: format_table_pct
            for column in PERCENT_STYLE_COLUMNS
            if column in data.columns
        }
    )
    formatters.update(
        {
            column: format_table_money
            for column in MONEY_STYLE_COLUMNS
            if column in data.columns
        }
    )
    formatters.update(
        {
            column: format_table_decimal
            for column in DECIMAL_STYLE_COLUMNS
            if column in data.columns
        }
    )
    if formatters:
        styler = styler.format(formatters)

    status_columns = [
        column for column in STATUS_STYLE_COLUMNS if column in data.columns
    ]
    signed_columns = [
        column for column in SIGNED_STYLE_COLUMNS if column in data.columns
    ]
    inverted_columns = [
        column
        for column in INVERTED_SIGNED_STYLE_COLUMNS
        if column in data.columns
    ]

    if status_columns:
        styler = styler.map(status_cell_style, subset=status_columns)
    if signed_columns:
        styler = styler.map(signed_number_style, subset=signed_columns)
    if inverted_columns:
        styler = styler.map(
            lambda value: signed_number_style(value, inverted=True),
            subset=inverted_columns,
        )
    return styler


def match_name(row: pd.Series) -> str:
    """Monta nome curto da partida."""
    return f"{row['HomeTeam']} x {row['AwayTeam']}"


def score_label(row: pd.Series) -> str:
    """Formata placar quando gols estao disponiveis."""
    if "FTHG" not in row or "FTAG" not in row:
        return ""
    if pd.isna(row["FTHG"]) or pd.isna(row["FTAG"]):
        return ""
    return f"{int(row['FTHG'])}-{int(row['FTAG'])}"


def apply_text_search(
    data: pd.DataFrame,
    query: str,
    columns: list[str],
) -> pd.DataFrame:
    """Filtra uma tabela por texto em colunas selecionadas."""
    if not query.strip():
        return data

    normalized = query.strip().casefold()
    mask = pd.Series(False, index=data.index)
    for column in columns:
        mask |= data[column].astype(str).str.casefold().str.contains(
            normalized,
            regex=False,
            na=False,
        )
    return data[mask].copy()


def sort_table(data: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    """Ordena tabelas de apostas e palpites por criterio visual."""
    if sort_mode == "Maior edge":
        return data.sort_values("EdgeTable", ascending=False)
    if sort_mode == "Maior prob.":
        return data.sort_values("ModelProb", ascending=False)
    if sort_mode == "Melhor odd":
        return data.sort_values("OddTable", ascending=False)
    if sort_mode == "Maior lucro" and "SimProfit" in data.columns:
        return data.sort_values("SimProfit", ascending=False)
    if sort_mode == "+EV primeiro" and "UiValueBet" in data.columns:
        return data.sort_values(
            ["UiValueBet", "MatchDatetime", "EdgeTable"],
            ascending=[False, True, False],
        )
    if sort_mode == "Proximos":
        return data.sort_values("MatchDatetime", ascending=True)
    return data.sort_values("MatchDatetime", ascending=False)


def render_pipeline_runner() -> None:
    """Renderiza controles para atualizar backtests pelo dashboard."""
    with st.sidebar.expander("Atualizar backtests", expanded=False):
        st.caption("Roda o pipeline local e atualiza os CSVs em outputs.")
        market = st.selectbox(
            "Mercados",
            options=["all", "over25", "under25", "result", "win"],
            format_func={
                "all": "Todos",
                "over25": "Over 2.5",
                "under25": "Under 2.5",
                "result": "Resultado 1X2",
                "win": "Vitoria Casa/Fora",
            }.get,
        )
        leagues = render_league_selector(
            "Ligas do treino",
            options=DEFAULT_LEAGUES,
            default=DEFAULT_LEAGUES,
            key="pipeline_train",
        )
        seasons = st.multiselect(
            "Temporadas",
            options=DEFAULT_SEASONS,
            default=DEFAULT_SEASONS,
        )
        feature_profile = st.segmented_control(
            "Perfil de features",
            options=["base", "extended"],
            default="base",
            format_func={
                "base": "Base",
                "extended": "Estendido",
            }.get,
            help="Estendido adiciona forma, descanso e aproveitamento recentes.",
            key="pipeline_feature_profile",
            width="stretch",
        )
        split_strategy = st.segmented_control(
            "Divisao temporal",
            options=["season", "chronological"],
            default="season",
            format_func={
                "season": "Por temporada",
                "chronological": "Cronologica",
            }.get,
            help=(
                "Por temporada usa temporadas inteiras por liga para "
                "treino, validacao e teste."
            ),
            key="pipeline_split_strategy",
            width="stretch",
        )
        walk_forward_splits = st.number_input(
            "Folds walk-forward",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Use 0 para uma atualizacao mais rapida.",
        )
        xgb_tuning_trials = st.number_input(
            "Tuning XGBoost",
            min_value=0,
            max_value=6,
            value=4,
            step=1,
            help="Numero de perfis testados em validacao temporal. 0 desativa.",
        )
        calibration_method = st.segmented_control(
            "Calibracao",
            options=["sigmoid", "isotonic"],
            default="sigmoid",
            format_func={
                "sigmoid": "Sigmoid",
                "isotonic": "Isotonic",
            }.get,
            help="Sigmoid e mais estavel; isotonic pode ajustar melhor com mais dados.",
            key="pipeline_calibration_method",
            width="stretch",
        )
        run_comparison = st.checkbox(
            "Comparar com odds vs sem odds",
            value=True,
        )
        run_optimization = st.checkbox(
            "Otimizar filtros por mercado",
            value=True,
        )
        run_realistic = st.checkbox(
            "Backtest realista",
            value=True,
            help="Gera validacao/teste por temporada, Kelly, drawdown e quebras.",
        )
        use_clubelo = st.checkbox(
            "Usar força real ClubElo",
            value=False,
            help=(
                "Baixa ratings historicos do ClubElo e usa Elo interno como "
                "fallback. A primeira execucao pode demorar mais."
            ),
        )

        run_clicked = st.button(
            "Rodar pipeline",
            type="primary",
            width="stretch",
        )

        if not run_clicked:
            return

        if not leagues or not seasons:
            st.error("Selecione ao menos uma liga e uma temporada.")
            return

        command = [
            sys.executable,
            str(PIPELINE_PATH),
            "--markets",
            market,
            "--leagues",
            *leagues,
            "--seasons",
            *seasons,
            "--feature-profile",
            feature_profile or "base",
            "--split-strategy",
            split_strategy or "season",
            "--validation-seasons",
            "1",
            "--test-seasons",
            "1",
            "--walk-forward-splits",
            str(walk_forward_splits),
            "--xgb-tuning-trials",
            str(xgb_tuning_trials),
            "--calibration-method",
            calibration_method or "sigmoid",
        ]
        if not run_comparison:
            command.append("--skip-model-comparison")
        if not run_optimization:
            command.append("--skip-filter-optimization")
        if not run_realistic:
            command.append("--skip-realistic-backtest")
        if use_clubelo:
            command.append("--use-clubelo")

        with st.spinner("Rodando pipeline. Isso pode levar alguns minutos..."):
            result = subprocess.run(
                command,
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=60 * 30,
                check=False,
            )

        log_text = "\n".join(
            part for part in [result.stdout, result.stderr] if part.strip()
        )
        st.session_state["last_pipeline_log"] = log_text[-8000:]

        if result.returncode != 0:
            st.error("Pipeline terminou com erro.")
            st.code(st.session_state["last_pipeline_log"], language="text")
            return

        st.cache_data.clear()
        st.success("Backtests atualizados.")
        st.code(st.session_state["last_pipeline_log"], language="text")


def render_upcoming_runner() -> None:
    """Renderiza controles para gerar palpites futuros."""
    with st.sidebar.expander("Atualizar palpites futuros", expanded=True):
        st.caption(
            "Usa fixtures publicas do Football-Data. Os palpites futuros "
            "ficam presos a Betano; se a fonte nao trouxer essa casa, a "
            "tela mostra isso com clareza."
        )
        market = st.selectbox(
            "Mercados dos palpites",
            options=["all", "over25", "under25", "result", "win"],
            format_func={
                "all": "Todos",
                "over25": "Over 2.5",
                "under25": "Under 2.5",
                "result": "Resultado 1X2",
                "win": "Vitoria Casa/Fora",
            }.get,
        )
        fixture_leagues = [league for league in DEFAULT_LEAGUES if league != "BRA"]
        leagues = render_league_selector(
            "Ligas dos palpites",
            options=fixture_leagues,
            default=fixture_leagues,
            key="upcoming_train",
        )
        seasons = st.multiselect(
            "Temporadas historicas",
            options=DEFAULT_SEASONS,
            default=DEFAULT_SEASONS,
        )
        feature_profile = st.segmented_control(
            "Perfil de features",
            options=["base", "extended"],
            default="base",
            format_func={
                "base": "Base",
                "extended": "Estendido",
            }.get,
            help="Estendido adiciona forma, descanso e aproveitamento recentes.",
            key="upcoming_feature_profile",
            width="stretch",
        )
        days_ahead = st.number_input(
            "Dias a frente",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
        )
        force_refresh = st.checkbox(
            "Forcar download das fixtures",
            value=True,
        )
        use_optimized_filters = st.checkbox(
            "Usar filtros otimizados positivos",
            value=True,
        )
        use_clubelo = st.checkbox(
            "Usar força real ClubElo",
            value=False,
            help=(
                "Baixa ratings historicos do ClubElo e usa Elo interno como "
                "fallback. A primeira execucao pode demorar mais."
            ),
            key="upcoming_use_clubelo",
        )
        xgb_tuning_trials = st.number_input(
            "Tuning XGBoost",
            min_value=0,
            max_value=6,
            value=4,
            step=1,
            help="Numero de perfis testados em validacao temporal. 0 desativa.",
        )
        run_clicked = st.button(
            "Gerar palpites",
            type="primary",
            width="stretch",
        )

        if not run_clicked:
            return

        if not leagues or not seasons:
            st.error("Selecione ao menos uma liga e uma temporada historica.")
            return

        command = [
            sys.executable,
            str(PREDICT_UPCOMING_PATH),
            "--markets",
            market,
            "--leagues",
            *leagues,
            "--seasons",
            *seasons,
            "--feature-profile",
            feature_profile or "base",
            "--days-ahead",
            str(days_ahead),
            "--xgb-tuning-trials",
            str(xgb_tuning_trials),
            "--preferred-bookmaker",
            "Betano",
        ]
        if force_refresh:
            command.append("--force-refresh-fixtures")
        if not use_optimized_filters:
            command.append("--ignore-optimized-filters")
        if use_clubelo:
            command.append("--use-clubelo")

        with st.spinner("Gerando palpites. O treino pode levar alguns minutos..."):
            result = subprocess.run(
                command,
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=60 * 30,
                check=False,
            )

        log_text = "\n".join(
            part for part in [result.stdout, result.stderr] if part.strip()
        )
        st.session_state["last_upcoming_log"] = log_text[-8000:]

        if result.returncode != 0:
            st.error("Geracao de palpites terminou com erro.")
            st.code(st.session_state["last_upcoming_log"], language="text")
            return

        st.cache_data.clear()
        st.success("Palpites futuros atualizados.")
        st.code(st.session_state["last_upcoming_log"], language="text")


def load_market_data(market: str) -> pd.DataFrame:
    """Carrega e padroniza colunas do mercado selecionado."""
    if market == "Over 2.5":
        if not OVER25_PATH.exists():
            st.error(f"Arquivo nao encontrado: {OVER25_PATH}")
            st.stop()

        data = load_csv(OVER25_PATH, OVER25_PATH.stat().st_mtime)
        data["Selection"] = "Over 2.5"
        data["SelectionOdd"] = data["Odd_Over25"]
        data["ModelProb"] = data["Model_Prob_Over25"]
        data["NoVigProb"] = data["NoVig_Prob_Over25"]
        data["MarketEdge"] = data["Edge"]
        data["Hit"] = data["Over25"].eq(1)
        data["DefaultBet"] = data["Bet_Over25"].astype(bool)
        data["Market"] = "Over 2.5"
        return data

    if market == "Under 2.5":
        if not UNDER25_PATH.exists():
            st.error(f"Arquivo nao encontrado: {UNDER25_PATH}")
            st.stop()

        data = load_csv(UNDER25_PATH, UNDER25_PATH.stat().st_mtime)
        data["Selection"] = "Under 2.5"
        data["SelectionOdd"] = data["Odd_Under25"]
        data["ModelProb"] = data["Model_Prob_Under25"]
        data["NoVigProb"] = data["NoVig_Prob_Under25"]
        data["MarketEdge"] = data["Under_Edge"]
        data["Hit"] = data["Over25"].eq(0)
        data["DefaultBet"] = data["Bet_Under25"].astype(bool)
        data["Market"] = "Under 2.5"
        return data

    if market == "Resultado 1X2":
        if not RESULT_1X2_PATH.exists():
            st.error(f"Arquivo nao encontrado: {RESULT_1X2_PATH}")
            st.stop()

        data = load_csv(RESULT_1X2_PATH, RESULT_1X2_PATH.stat().st_mtime)
        data["Selection"] = data["Best_Result_Name"]
        data["SelectionOdd"] = data["Best_Odd"]
        data["ModelProb"] = data["Best_Model_Prob"]
        data["NoVigProb"] = data["Best_NoVig_Prob"]
        data["MarketEdge"] = data["Best_Edge"]
        data["Hit"] = data["ResultTarget"].eq(data["Best_Result_Index"])
        data["DefaultBet"] = data["Bet_Result"].astype(bool)
        data["Market"] = "Resultado 1X2"
        return data

    if not WIN_PATH.exists():
        st.error(f"Arquivo nao encontrado: {WIN_PATH}")
        st.stop()

    data = load_csv(WIN_PATH, WIN_PATH.stat().st_mtime)
    data["Selection"] = data["Win_Result_Name"]
    data["SelectionOdd"] = data["Win_Odd"]
    data["ModelProb"] = data["Win_Model_Prob"]
    data["NoVigProb"] = data["Win_NoVig_Prob"]
    data["MarketEdge"] = data["Win_Edge"]
    data["Hit"] = data["ResultTarget"].eq(data["Win_Result_Index"])
    data["DefaultBet"] = data["Bet_Win"].astype(bool)
    data["Market"] = "Vitoria Casa/Fora"
    return data


def load_available_backtests() -> pd.DataFrame:
    """Carrega todos os mercados historicos que possuem CSV disponivel."""
    frames = []
    for market in BACKTEST_MARKETS:
        path = BACKTEST_MARKET_PATHS[market]
        if path.exists():
            frames.append(load_market_data(market))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def apply_sidebar_filters(data: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Aplica filtros de data e liga."""
    st.sidebar.header("Filtros")

    min_date = data["MatchDate"].min()
    max_date = data["MatchDate"].max()
    start_date, end_date = st.sidebar.date_input(
        "Periodo",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    leagues = ordered_leagues(data["Liga"].dropna().unique().tolist())
    selected_leagues = render_league_selector(
        "Ligas",
        options=leagues,
        default=leagues,
        key=f"{key_prefix}_leagues",
        sidebar=True,
    )

    filtered = data[
        data["MatchDate"].between(start_date, end_date)
        & data["Liga"].isin(selected_leagues)
    ].copy()
    return filtered


def simulate_bets(
    data: pd.DataFrame,
    edge: float,
    min_probability: float,
    max_odd: float,
    stake: float,
) -> pd.DataFrame:
    """Recalcula apostas simuladas com os filtros escolhidos na UI."""
    simulated = data.copy()
    odd_filter = True if max_odd <= 0 else simulated["SelectionOdd"] <= max_odd
    simulated["SimBet"] = (
        simulated["MarketEdge"].ge(edge)
        & simulated["ModelProb"].ge(min_probability)
        & odd_filter
    )
    simulated["SimStake"] = simulated["SimBet"].astype(float) * stake
    simulated["SimProfit"] = 0.0
    simulated.loc[simulated["SimBet"] & simulated["Hit"], "SimProfit"] = (
        stake
        * (
            simulated.loc[
                simulated["SimBet"] & simulated["Hit"],
                "SelectionOdd",
            ]
            - 1.0
        )
    )
    simulated.loc[simulated["SimBet"] & ~simulated["Hit"], "SimProfit"] = -stake
    simulated["CumulativeProfit"] = simulated["SimProfit"].cumsum()
    return simulated


def build_summary(data: pd.DataFrame) -> dict[str, float]:
    """Calcula resumo financeiro e operacional."""
    bets = data[data["SimBet"]].copy()
    total_bets = len(bets)
    total_staked = float(bets["SimStake"].sum())
    total_profit = float(bets["SimProfit"].sum())
    roi = total_profit / total_staked if total_staked else 0.0
    hit_rate = float(bets["Hit"].mean()) if total_bets else 0.0
    avg_odd = float(bets["SelectionOdd"].mean()) if total_bets else 0.0
    avg_prob = float(bets["ModelProb"].mean()) if total_bets else 0.0
    avg_edge = float(bets["MarketEdge"].mean()) if total_bets else 0.0

    return {
        "bets": total_bets,
        "staked": total_staked,
        "profit": total_profit,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_odd": avg_odd,
        "avg_prob": avg_prob,
        "avg_edge": avg_edge,
    }


def render_metric_row(summary: dict[str, float]) -> None:
    """Mostra metricas principais em linha."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Apostas", f"{summary['bets']:,}".replace(",", "."))
    col2.metric("Lucro", format_money(summary["profit"]))
    col3.metric("ROI", format_pct(summary["roi"]))
    col4.metric("Acerto", format_pct(summary["hit_rate"]))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total apostado", format_money(summary["staked"]))
    col6.metric("Odd media", f"{summary['avg_odd']:.2f}")
    col7.metric("Prob. media", format_pct(summary["avg_prob"]))
    col8.metric("Edge medio", format_pct(summary["avg_edge"]))


def render_charts(data: pd.DataFrame) -> None:
    """Renderiza graficos principais."""
    bets = data[data["SimBet"]].copy()

    if bets.empty:
        st.info("Nenhuma aposta encontrada com os filtros atuais.")
        return

    left, right = st.columns((1.4, 1.0))
    with left:
        profit_curve = bets.sort_values("MatchDatetime").copy()
        profit_curve["CumulativeProfit"] = profit_curve["SimProfit"].cumsum()
        fig = px.line(
            profit_curve,
            x="MatchDatetime",
            y="CumulativeProfit",
            color="Market",
            labels={
                "MatchDatetime": "Data",
                "CumulativeProfit": "Lucro acumulado",
                "Market": "Mercado",
            },
            title="Lucro acumulado",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        league_summary = (
            bets.groupby("Liga", as_index=False)
            .agg(
                Apostas=("SimBet", "size"),
                Lucro=("SimProfit", "sum"),
                Apostado=("SimStake", "sum"),
                Acerto=("Hit", "mean"),
            )
            .sort_values("Lucro", ascending=False)
        )
        league_summary["ROI"] = league_summary["Lucro"] / league_summary["Apostado"]
        league_summary["LigaNome"] = league_summary["Liga"].map(league_label)
        fig = px.bar(
            league_summary,
            x="LigaNome",
            y="Lucro",
            color="ROI",
            hover_data=["Liga", "Apostas", "Acerto", "ROI"],
            labels={"LigaNome": "Liga", "Lucro": "Lucro"},
            title="Lucro por liga",
        )
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            bets,
            x="SelectionOdd",
            nbins=24,
            color="Hit",
            labels={"SelectionOdd": "Odd", "count": "Apostas", "Hit": "Acertou"},
            title="Distribuicao das odds",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.scatter(
            bets,
            x="ModelProb",
            y="MarketEdge",
            color="Hit",
            size="SelectionOdd",
            hover_data=[
                "LigaNome",
                "HomeTeam",
                "AwayTeam",
                "Selection",
                "SelectionOdd",
            ],
            labels={
                "ModelProb": "Probabilidade do modelo",
                "MarketEdge": "Edge",
                "Hit": "Acertou",
            },
            title="Probabilidade vs. edge",
        )
        st.plotly_chart(fig, width="stretch")


def render_league_summary_table(data: pd.DataFrame) -> None:
    """Mostra uma tabela resumida das apostas por liga."""
    bets = data[data["SimBet"]].copy()
    if bets.empty:
        return

    summary = (
        bets.groupby("LigaNome", as_index=False)
        .agg(
            Apostas=("SimBet", "size"),
            Lucro=("SimProfit", "sum"),
            Apostado=("SimStake", "sum"),
            Acerto=("Hit", "mean"),
            OddMedia=("SelectionOdd", "mean"),
            EdgeMedio=("MarketEdge", "mean"),
        )
        .sort_values("Lucro", ascending=False)
    )
    summary["ROI"] = summary["Lucro"] / summary["Apostado"]

    st.subheader("Resumo por liga")
    st.dataframe(
        style_dashboard_table(summary),
        width="stretch",
        hide_index=True,
        column_config={
            "LigaNome": "Liga",
            "Apostas": st.column_config.NumberColumn("Apostas", format="%.0f"),
            "Lucro": st.column_config.NumberColumn("Lucro", format="R$ %.2f"),
            "Apostado": st.column_config.NumberColumn("Apostado", format="R$ %.2f"),
            "Acerto": st.column_config.NumberColumn("Acerto", format="%.3f"),
            "OddMedia": st.column_config.NumberColumn("Odd media", format="%.2f"),
            "EdgeMedio": st.column_config.NumberColumn("Edge medio", format="%.3f"),
            "ROI": st.column_config.NumberColumn("ROI", format="%.3f"),
        },
    )


def render_table(data: pd.DataFrame) -> None:
    """Mostra tabela das apostas selecionadas."""
    bets = data[data["SimBet"]].copy()
    if bets.empty:
        return

    st.subheader("Apostas simuladas")
    col1, col2, col3 = st.columns((1.3, 1.0, 0.8))
    with col1:
        sort_mode = st.segmented_control(
            "Ordenar apostas",
            options=[
                "Recentes",
                "Maior edge",
                "Maior prob.",
                "Melhor odd",
                "Maior lucro",
            ],
            default="Recentes",
            key="backtest_table_sort",
            width="stretch",
        )
    with col2:
        outcome = st.pills(
            "Status",
            options=["Todos", "Green", "Red"],
            default="Todos",
            key="backtest_table_status",
            width="stretch",
        )
    with col3:
        row_limit = st.selectbox(
            "Linhas",
            options=[25, 50, 100, 250, 500],
            index=2,
            key="backtest_table_rows",
        )
    query = st.text_input(
        "Buscar time ou liga",
        value="",
        key="backtest_table_search",
    )

    table = bets.copy()
    table["Jogo"] = table.apply(match_name, axis=1)
    table["Placar"] = table.apply(score_label, axis=1)
    table["StatusAposta"] = table["Hit"].map({True: "Green", False: "Red"})
    table["ValueGap"] = table["ModelProb"] - table["NoVigProb"]
    table["EdgeTable"] = table["MarketEdge"]
    table["OddTable"] = table["SelectionOdd"]
    if outcome != "Todos":
        table = table[table["StatusAposta"].eq(outcome)].copy()
    table = apply_text_search(
        table,
        query,
        ["LigaNome", "HomeTeam", "AwayTeam", "Selection", "Jogo"],
    )
    table = sort_table(table, sort_mode).head(row_limit)
    table = table[
        [
            "MatchDatetime",
            "LigaNome",
            "Jogo",
            "Placar",
            "Selection",
            "SelectionOdd",
            "ModelProb",
            "NoVigProb",
            "ValueGap",
            "MarketEdge",
            "StatusAposta",
            "SimProfit",
        ]
    ]

    st.dataframe(
        style_dashboard_table(table),
        width="stretch",
        hide_index=True,
        column_config={
            "MatchDatetime": st.column_config.DatetimeColumn("Data"),
            "LigaNome": "Liga",
            "Jogo": "Jogo",
            "Placar": "Placar",
            "Selection": "Selecao",
            "SelectionOdd": st.column_config.NumberColumn("Odd", format="%.2f"),
            "ModelProb": st.column_config.ProgressColumn(
                "Prob. modelo",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "NoVigProb": st.column_config.ProgressColumn(
                "Prob. no-vig",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "ValueGap": st.column_config.NumberColumn(
                "Modelo - no-vig",
                format="%.3f",
            ),
            "MarketEdge": st.column_config.NumberColumn("Edge", format="%.3f"),
            "StatusAposta": "Status",
            "SimProfit": st.column_config.NumberColumn("Lucro", format="R$ %.2f"),
        },
    )


def add_strength_segments(data: pd.DataFrame) -> pd.DataFrame:
    """Cria faixas legiveis para descobrir onde o modelo performa melhor."""
    segmented = data.copy()
    segmented["CotacaoFaixa"] = pd.cut(
        pd.to_numeric(segmented["SelectionOdd"], errors="coerce"),
        bins=[0.0, 1.50, 1.80, 2.20, 3.00, 5.00, np.inf],
        labels=[
            "Ate 1.50",
            "1.51 a 1.80",
            "1.81 a 2.20",
            "2.21 a 3.00",
            "3.01 a 5.00",
            "Acima de 5.00",
        ],
        include_lowest=True,
    )
    segmented["ChanceFaixa"] = pd.cut(
        pd.to_numeric(segmented["ModelProb"], errors="coerce"),
        bins=[0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0],
        labels=[
            "Ate 45%",
            "45% a 50%",
            "50% a 55%",
            "55% a 60%",
            "60% a 65%",
            "65% a 70%",
            "Acima de 70%",
        ],
        include_lowest=True,
    )
    segmented["VantagemFaixa"] = pd.cut(
        pd.to_numeric(segmented["MarketEdge"], errors="coerce"),
        bins=[-np.inf, 0.0, 0.02, 0.05, 0.08, 0.12, np.inf],
        labels=[
            "Sem vantagem",
            "0% a 2%",
            "2% a 5%",
            "5% a 8%",
            "8% a 12%",
            "Acima de 12%",
        ],
    )

    if "MatchImportance" in segmented.columns:
        segmented["ImportanciaFaixa"] = pd.cut(
            pd.to_numeric(segmented["MatchImportance"], errors="coerce"),
            bins=[-0.01, 0.33, 0.66, 1.0],
            labels=["Baixa", "Media", "Alta"],
        )
    if "xG_Expected_Total_Match_Roll5" in segmented.columns:
        segmented["GolsEsperadosFaixa"] = pd.cut(
            pd.to_numeric(
                segmented["xG_Expected_Total_Match_Roll5"],
                errors="coerce",
            ),
            bins=[-np.inf, 2.00, 2.50, 3.00, 3.50, np.inf],
            labels=[
                "Ate 2.00",
                "2.01 a 2.50",
                "2.51 a 3.00",
                "3.01 a 3.50",
                "Acima de 3.50",
            ],
        )
    if "LineupStrength_Diff" in segmented.columns:
        segmented["ForcaTitularesFaixa"] = pd.cut(
            pd.to_numeric(segmented["LineupStrength_Diff"], errors="coerce"),
            bins=[-np.inf, -0.15, 0.15, np.inf],
            labels=["Visitante melhor", "Equilibrado", "Casa melhor"],
        )
    return segmented


def strength_status(row: pd.Series, min_bets: int) -> str:
    """Classifica um recorte pelo retorno, acerto e volume."""
    if row["Apostas"] < min_bets:
        return "Poucos jogos"
    if row["Retorno"] > 0 and row["SobraAcerto"] > 0:
        return "Forte"
    if row["Retorno"] > 0:
        return "Promissor"
    if row["SobraAcerto"] > 0:
        return "Instavel"
    return "Evitar"


def summarize_strength_groups(
    data: pd.DataFrame,
    group_columns: list[str],
    label: str,
    min_bets: int,
) -> pd.DataFrame:
    """Resume retorno e acerto para um recorte especifico."""
    bets = data[data["SimBet"]].copy()
    group_columns = [column for column in group_columns if column in bets.columns]
    if bets.empty or not group_columns:
        return pd.DataFrame()

    grouped = bets.groupby(group_columns, dropna=False, observed=True)
    summary = grouped.agg(
        Apostas=("SimBet", "size"),
        Lucro=("SimProfit", "sum"),
        Apostado=("SimStake", "sum"),
        Acerto=("Hit", "mean"),
        CotacaoMedia=("SelectionOdd", "mean"),
        ChanceMedia=("ModelProb", "mean"),
        VantagemMedia=("MarketEdge", "mean"),
        ChanceMercado=("NoVigProb", "mean"),
    ).reset_index()
    if summary.empty:
        return summary

    if len(group_columns) == 1:
        summary["Grupo"] = summary[group_columns[0]].astype(str)
    else:
        summary["Grupo"] = summary[group_columns].astype(str).agg(" / ".join, axis=1)
    summary["Grupo"] = summary["Grupo"].replace({"nan": "Sem dado"})
    summary["Recorte"] = label
    summary["Retorno"] = np.where(
        summary["Apostado"].gt(0),
        summary["Lucro"] / summary["Apostado"],
        0.0,
    )
    summary["AcertoNecessario"] = summary["ChanceMercado"].fillna(
        1.0 / summary["CotacaoMedia"].replace(0, np.nan),
    )
    summary["SobraAcerto"] = summary["Acerto"] - summary["AcertoNecessario"]
    summary["LucroPorAposta"] = np.where(
        summary["Apostas"].gt(0),
        summary["Lucro"] / summary["Apostas"],
        0.0,
    )
    volume_weight = np.minimum(1.0, summary["Apostas"] / max(min_bets, 1))
    summary["Nota"] = (summary["Retorno"] + summary["SobraAcerto"]) * volume_weight
    summary["Leitura"] = summary.apply(
        lambda row: strength_status(row, min_bets),
        axis=1,
    )
    return summary.sort_values(
        ["Nota", "Lucro", "Apostas"],
        ascending=[False, False, False],
        kind="mergesort",
    )


def strength_column_config() -> dict[str, object]:
    """Configuracao comum das tabelas de pontos fortes."""
    return {
        "Recorte": "Recorte",
        "Grupo": "Grupo",
        "Leitura": "Leitura",
        "Apostas": st.column_config.NumberColumn("Apostas", format="%.0f"),
        "Lucro": st.column_config.NumberColumn("Lucro", format="R$ %.2f"),
        "Apostado": st.column_config.NumberColumn("Apostado", format="R$ %.2f"),
        "Retorno": st.column_config.NumberColumn("Retorno", format="percent"),
        "Acerto": st.column_config.NumberColumn("Acerto", format="percent"),
        "AcertoNecessario": st.column_config.NumberColumn(
            "Acerto minimo",
            format="percent",
        ),
        "SobraAcerto": st.column_config.NumberColumn(
            "Sobra no acerto",
            format="percent",
        ),
        "CotacaoMedia": st.column_config.NumberColumn(
            "Cotacao media",
            format="%.2f",
        ),
        "ChanceMedia": st.column_config.NumberColumn(
            "Chance media",
            format="percent",
        ),
        "VantagemMedia": st.column_config.NumberColumn(
            "Vantagem media",
            format="percent",
        ),
        "LucroPorAposta": st.column_config.NumberColumn(
            "Lucro/aposta",
            format="R$ %.2f",
        ),
        "Nota": st.column_config.NumberColumn("Nota", format="%.3f"),
    }


def render_strength_table(summary: pd.DataFrame, row_limit: int = 80) -> None:
    """Mostra tabela padronizada de recortes."""
    if summary.empty:
        st.info("Nenhum recorte encontrado com os filtros atuais.")
        return

    visible_columns = [
        "Recorte",
        "Grupo",
        "Leitura",
        "Apostas",
        "Lucro",
        "Retorno",
        "Acerto",
        "AcertoNecessario",
        "SobraAcerto",
        "CotacaoMedia",
        "ChanceMedia",
        "VantagemMedia",
        "LucroPorAposta",
        "Nota",
    ]
    visible_columns = [column for column in visible_columns if column in summary]
    st.dataframe(
        style_dashboard_table(summary[visible_columns].head(row_limit)),
        width="stretch",
        hide_index=True,
        column_config=strength_column_config(),
    )


def render_strength_charts(summary: pd.DataFrame, title: str) -> None:
    """Mostra graficos para comparar recortes fortes e fracos."""
    if summary.empty:
        return

    plot_data = summary.head(25).copy()
    left, right = st.columns((1.25, 1.0))
    with left:
        fig = px.bar(
            plot_data.sort_values("Retorno", ascending=True),
            x="Retorno",
            y="Grupo",
            color="Retorno",
            hover_data=["Recorte", "Apostas", "Lucro", "Acerto", "CotacaoMedia"],
            labels={
                "Grupo": "Grupo",
                "Retorno": "Retorno",
                "Apostas": "Apostas",
                "Lucro": "Lucro",
                "Acerto": "Acerto",
                "CotacaoMedia": "Cotacao media",
            },
            title=title,
            color_continuous_scale="RdYlGn",
        )
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.scatter(
            summary,
            x="Acerto",
            y="Retorno",
            size="Apostas",
            color="Leitura",
            hover_data=["Recorte", "Grupo", "Lucro", "SobraAcerto", "CotacaoMedia"],
            labels={
                "Acerto": "Acerto",
                "Retorno": "Retorno",
                "Apostas": "Apostas",
                "Leitura": "Leitura",
                "SobraAcerto": "Sobra no acerto",
                "CotacaoMedia": "Cotacao media",
            },
            title="Acerto vs retorno",
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, width="stretch")


def build_strength_dimensions(data: pd.DataFrame) -> dict[str, list[str]]:
    """Define recortes disponiveis conforme as colunas carregadas."""
    dimensions = {
        "Campeonato": ["LigaNome"],
        "Campeonato + mercado": ["LigaNome", "Market"],
        "Mercado": ["Market"],
        "Palpite": ["Selection"],
        "Mercado + palpite": ["Market", "Selection"],
        "Cotacao": ["CotacaoFaixa"],
        "Chance do modelo": ["ChanceFaixa"],
        "Vantagem": ["VantagemFaixa"],
    }
    optional_dimensions = {
        "Importancia do jogo": ["ImportanciaFaixa"],
        "Gols esperados": ["GolsEsperadosFaixa"],
        "Forca dos titulares": ["ForcaTitularesFaixa"],
    }
    dimensions.update(
        {
            label: columns
            for label, columns in optional_dimensions.items()
            if all(column in data.columns for column in columns)
        }
    )
    return dimensions


def build_strength_summaries(
    data: pd.DataFrame,
    min_bets: int,
) -> dict[str, pd.DataFrame]:
    """Monta resumos para todos os recortes do diagnostico."""
    dimensions = build_strength_dimensions(data)
    return {
        label: summarize_strength_groups(data, columns, label, min_bets)
        for label, columns in dimensions.items()
    }


def build_rule_suggestions(
    data: pd.DataFrame,
    min_bets: int,
    stake: float,
) -> pd.DataFrame:
    """Testa combinacoes simples para sugerir regras de entrada."""
    if data.empty:
        return pd.DataFrame()

    rows = []
    edge_options = [0.00, 0.02, 0.05, 0.08, 0.10, 0.12]
    probability_options = [0.45, 0.50, 0.55, 0.60, 0.65]
    max_odd_options = [0.0, 1.60, 1.80, 2.20, 2.50, 3.00, 5.00]

    for market, market_data in data.groupby("Market", sort=False):
        for edge in edge_options:
            for min_probability in probability_options:
                for max_odd in max_odd_options:
                    simulated = simulate_bets(
                        market_data,
                        edge=edge,
                        min_probability=min_probability,
                        max_odd=max_odd,
                        stake=stake,
                    )
                    bets = simulated[simulated["SimBet"]].copy()
                    if len(bets) < min_bets:
                        continue

                    summary = build_summary(simulated)
                    break_even = float(
                        (1.0 / bets["SelectionOdd"].replace(0, np.nan)).mean()
                    )
                    row = {
                        "Market": market,
                        "Apostas": summary["bets"],
                        "Lucro": summary["profit"],
                        "Retorno": summary["roi"],
                        "Acerto": summary["hit_rate"],
                        "AcertoNecessario": break_even,
                        "SobraAcerto": summary["hit_rate"] - break_even,
                        "CotacaoMedia": summary["avg_odd"],
                        "ChanceMedia": summary["avg_prob"],
                        "VantagemMedia": summary["avg_edge"],
                    }
                    row["LucroPorAposta"] = row["Lucro"] / row["Apostas"]
                    row["Nota"] = row["Retorno"] + row["SobraAcerto"]
                    row["Leitura"] = strength_status(pd.Series(row), min_bets)
                    row["Regra"] = (
                        f"Vantagem {edge:.0%}+ | Chance {min_probability:.0%}+ | "
                        f"Cotacao {'sem limite' if max_odd <= 0 else f'ate {max_odd:.2f}'}"
                    )
                    rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["Nota", "Lucro", "Apostas"],
        ascending=[False, False, False],
        kind="mergesort",
    )


def render_rule_suggestions(data: pd.DataFrame, min_bets: int, stake: float) -> None:
    """Mostra regras candidatas para melhorar acerto e retorno."""
    suggestions = build_rule_suggestions(data, min_bets=min_bets, stake=stake)
    if suggestions.empty:
        st.info("Nenhuma regra teve jogos suficientes com os filtros atuais.")
        return

    visible_columns = [
        "Leitura",
        "Market",
        "Regra",
        "Apostas",
        "Lucro",
        "Retorno",
        "Acerto",
        "AcertoNecessario",
        "SobraAcerto",
        "CotacaoMedia",
        "ChanceMedia",
        "VantagemMedia",
        "LucroPorAposta",
        "Nota",
    ]
    st.dataframe(
        style_dashboard_table(suggestions[visible_columns].head(120)),
        width="stretch",
        hide_index=True,
        column_config={
            **strength_column_config(),
            "Market": "Mercado",
            "Regra": "Regra sugerida",
        },
    )


def apply_strength_filters(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Aplica filtros da tela de pontos fortes."""
    st.sidebar.header("Filtros")
    data_markets = set(data["Market"].unique())
    available_markets = [
        market for market in BACKTEST_MARKETS if market in data_markets
    ]
    selected_markets = st.sidebar.multiselect(
        "Mercados",
        options=available_markets,
        default=available_markets,
    )

    leagues = ordered_leagues(data["Liga"].dropna().unique().tolist())
    selected_leagues = render_league_selector(
        "Campeonatos",
        options=leagues,
        default=leagues,
        key="strength_leagues",
        sidebar=True,
    )

    min_date = data["MatchDate"].min()
    max_date = data["MatchDate"].max()
    selected_dates = st.sidebar.date_input(
        "Periodo",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="strength_period",
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    st.sidebar.header("Regra base")
    edge = st.sidebar.slider(
        "Vantagem minima",
        min_value=0.0,
        max_value=0.20,
        value=0.05,
        step=0.005,
        format="%.3f",
        key="strength_edge",
    )
    min_probability = st.sidebar.slider(
        "Chance minima",
        min_value=0.0,
        max_value=0.90,
        value=0.50,
        step=0.01,
        format="%.2f",
        key="strength_probability",
    )
    max_odd = st.sidebar.number_input(
        "Cotacao maxima (0 desativa)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.05,
        key="strength_max_odd",
    )
    stake = st.sidebar.number_input(
        "Valor da aposta",
        min_value=1.0,
        max_value=1000.0,
        value=10.0,
        step=1.0,
        key="strength_stake",
    )
    min_bets = st.sidebar.number_input(
        "Min. apostas por grupo",
        min_value=5,
        max_value=500,
        value=25,
        step=5,
        key="strength_min_bets",
    )

    filtered = data[
        data["Market"].isin(selected_markets)
        & data["Liga"].isin(selected_leagues)
        & data["MatchDate"].between(start_date, end_date)
    ].copy()
    params = {
        "edge": edge,
        "min_probability": min_probability,
        "max_odd": max_odd,
        "stake": stake,
        "min_bets": int(min_bets),
    }
    return filtered, params


def render_model_strength_page() -> None:
    """Mostra onde o modelo historicamente gera mais acerto e retorno."""
    st.title("Apostasbot | Onde funciona melhor")
    st.caption(
        "Descubra os recortes em que o modelo mais gera lucro, acerto e "
        "vantagem real sobre o mercado."
    )
    render_pipeline_runner()

    data = load_available_backtests()
    if data.empty:
        st.info("Rode a atualizacao dos testes historicos para gerar os CSVs.")
        return

    filtered, params = apply_strength_filters(data)
    if filtered.empty:
        st.info("Nenhum jogo encontrado com os filtros atuais.")
        return

    simulated = simulate_bets(
        filtered.sort_values("MatchDatetime"),
        edge=float(params["edge"]),
        min_probability=float(params["min_probability"]),
        max_odd=float(params["max_odd"]),
        stake=float(params["stake"]),
    )
    if not simulated["SimBet"].any():
        st.info("A regra base nao gerou apostas neste periodo.")
        return

    render_metric_row(build_summary(simulated))
    st.divider()

    segmented = add_strength_segments(simulated)
    summaries = build_strength_summaries(
        segmented,
        min_bets=int(params["min_bets"]),
    )
    all_summaries = [
        summary for summary in summaries.values() if not summary.empty
    ]
    combined = (
        pd.concat(all_summaries, ignore_index=True, sort=False)
        if all_summaries
        else pd.DataFrame()
    )
    qualified = (
        combined[combined["Apostas"].ge(int(params["min_bets"]))].copy()
        if not combined.empty
        else combined
    )

    st.subheader("Melhores recortes")
    best_groups = qualified if not qualified.empty else combined
    render_strength_charts(best_groups.head(50), "Top recortes por retorno")
    render_strength_table(best_groups, row_limit=40)

    tabs = st.tabs(
        [
            "Campeonatos",
            "Faixas",
            "Mercado e palpite",
            "Regras sugeridas",
            "Todos os recortes",
        ]
    )
    with tabs[0]:
        league_summary = summaries.get("Campeonato", pd.DataFrame())
        render_strength_charts(league_summary, "Retorno por campeonato")
        render_strength_table(league_summary)
        league_market = summaries.get("Campeonato + mercado", pd.DataFrame())
        st.subheader("Campeonato + mercado")
        render_strength_table(league_market, row_limit=120)

    with tabs[1]:
        faixa_options = [
            label
            for label in [
                "Cotacao",
                "Chance do modelo",
                "Vantagem",
                "Importancia do jogo",
                "Gols esperados",
                "Forca dos titulares",
            ]
            if label in summaries
        ]
        selected_faixa = st.selectbox(
            "Ver faixa",
            options=faixa_options,
            key="strength_band_select",
        )
        faixa_summary = summaries.get(selected_faixa, pd.DataFrame())
        render_strength_charts(faixa_summary, f"Retorno por {selected_faixa.lower()}")
        render_strength_table(faixa_summary)

    with tabs[2]:
        market_summary = summaries.get("Mercado", pd.DataFrame())
        selection_summary = summaries.get("Mercado + palpite", pd.DataFrame())
        render_strength_charts(market_summary, "Retorno por mercado")
        render_strength_table(market_summary)
        st.subheader("Mercado + palpite")
        render_strength_table(selection_summary)

    with tabs[3]:
        render_rule_suggestions(
            filtered.sort_values("MatchDatetime"),
            min_bets=int(params["min_bets"]),
            stake=float(params["stake"]),
        )

    with tabs[4]:
        render_strength_table(best_groups, row_limit=250)


def load_upcoming_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega palpites futuros, odds por casa e contexto da casa usada."""
    if not UPCOMING_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    predictions = load_optional_csv(UPCOMING_PATH, UPCOMING_PATH.stat().st_mtime)
    if not predictions.empty:
        defaults = {
            "RuleSource": "Legado",
            "RuleEdgeThreshold": pd.NA,
            "RuleMinModelProb": pd.NA,
            "RuleMaxOdd": pd.NA,
            "RuleEvalROI": pd.NA,
            "RuleEvalBets": pd.NA,
            "Home_PreMatch_Rank": pd.NA,
            "Away_PreMatch_Rank": pd.NA,
            "MatchImportance": pd.NA,
            "Home_MatchImportance": pd.NA,
            "Away_MatchImportance": pd.NA,
            "SeasonProgress": pd.NA,
            "Home_LineupDataAvailable": pd.NA,
            "Away_LineupDataAvailable": pd.NA,
            "Home_LineupConfirmed": pd.NA,
            "Away_LineupConfirmed": pd.NA,
            "LineupStrength_Diff": pd.NA,
            "MissingStarters_Diff": pd.NA,
            "MissingKeyPlayers_Diff": pd.NA,
            "Home_xGAvailable_Roll5": pd.NA,
            "Away_xGAvailable_Roll5": pd.NA,
            "Home_xGFor_Roll5": pd.NA,
            "Away_xGFor_Roll5": pd.NA,
            "xG_Total_Roll5": pd.NA,
            "xG_Expected_Total_Match_Roll5": pd.NA,
            "xG_Diff_Roll5": pd.NA,
            "RequestedBookmaker": "Melhor disponivel",
            "PreferredBookmakerAvailable": pd.NA,
        }
        for column, default_value in defaults.items():
            if column not in predictions.columns:
                predictions[column] = default_value

    if UPCOMING_ODDS_PATH.exists():
        odds = load_optional_csv(
            UPCOMING_ODDS_PATH,
            UPCOMING_ODDS_PATH.stat().st_mtime,
        )
    else:
        odds = pd.DataFrame()
    if UPCOMING_CONTEXT_PATH.exists():
        context = load_optional_csv(
            UPCOMING_CONTEXT_PATH,
            UPCOMING_CONTEXT_PATH.stat().st_mtime,
        )
    else:
        context = pd.DataFrame()
    return predictions, odds, context


def use_generated_upcoming_value(data: pd.DataFrame) -> pd.DataFrame:
    """Usa a flag +EV calculada pelo script de palpites."""
    generated = data.copy()
    generated["UiValueBet"] = generated["IsValueBet"].astype(bool)
    return generated


def recalculate_upcoming_value(
    data: pd.DataFrame,
    edge: float,
    min_probability: float,
    max_odd: float,
) -> pd.DataFrame:
    """Recalcula flags +EV dos palpites futuros com filtros da UI."""
    recalculated = data.copy()
    odd_filter = True if max_odd <= 0 else recalculated["BestOdd"] <= max_odd
    recalculated["UiValueBet"] = (
        recalculated["Edge"].ge(edge)
        & recalculated["ModelProb"].ge(min_probability)
        & odd_filter
    )
    return recalculated


def apply_upcoming_filters(data: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtros especificos dos palpites futuros."""
    if data.empty:
        return data

    st.sidebar.header("Filtros dos palpites")
    markets = sorted(data["Market"].dropna().unique().tolist())
    selected_markets = st.sidebar.multiselect(
        "Mercados",
        options=markets,
        default=markets,
    )

    leagues = ordered_leagues(data["Liga"].dropna().unique().tolist())
    selected_leagues = render_league_selector(
        "Ligas dos palpites",
        options=leagues,
        default=leagues,
        key="upcoming_filter",
        sidebar=True,
    )

    min_date = data["MatchDate"].min()
    max_date = data["MatchDate"].max()
    start_date, end_date = st.sidebar.date_input(
        "Periodo dos jogos",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    only_value = st.sidebar.checkbox("Mostrar apenas +EV", value=False)

    filtered = data[
        data["Market"].isin(selected_markets)
        & data["Liga"].isin(selected_leagues)
        & data["MatchDate"].between(start_date, end_date)
    ].copy()
    if only_value:
        filtered = filtered[filtered["UiValueBet"]].copy()
    return filtered


def render_upcoming_metrics(data: pd.DataFrame) -> None:
    """Mostra metricas resumidas dos palpites futuros."""
    value_bets = data[data["UiValueBet"]].copy()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Palpites", f"{len(data):,}".replace(",", "."))
    col2.metric("+EV", f"{len(value_bets):,}".replace(",", "."))
    col3.metric(
        "Edge medio +EV",
        format_pct(float(value_bets["Edge"].mean()) if len(value_bets) else 0.0),
    )
    col4.metric(
        "Odd media +EV",
        f"{float(value_bets['BestOdd'].mean()):.2f}" if len(value_bets) else "0.00",
    )


def rule_filter_label(row: pd.Series) -> str:
    """Resume a regra usada para classificar +EV."""
    edge = row.get("RuleEdgeThreshold")
    probability = row.get("RuleMinModelProb")
    max_odd = row.get("RuleMaxOdd")
    if pd.isna(edge) and pd.isna(probability) and pd.isna(max_odd):
        return "-"

    edge_text = "-" if pd.isna(edge) else f"{float(edge):.1%}"
    prob_text = "-" if pd.isna(probability) else f"{float(probability):.1%}"
    odd_text = "Sem limite" if pd.isna(max_odd) else f"{float(max_odd):.2f}"
    return f"Edge {edge_text} | Prob {prob_text} | Odd {odd_text}"


def importance_label(value: float) -> str:
    """Classifica importancia do jogo em leitura curta."""
    if pd.isna(value):
        return "-"
    if value >= 0.66:
        return "Alta"
    if value >= 0.33:
        return "Media"
    return "Baixa"


def lineup_label(row: pd.Series) -> str:
    """Resume se ha dados de escalação para o jogo."""
    def numeric_or_zero(value: object) -> float:
        if pd.isna(value):
            return 0.0
        return float(value)

    home_available = row.get("Home_LineupDataAvailable")
    away_available = row.get("Away_LineupDataAvailable")
    if pd.isna(home_available) and pd.isna(away_available):
        return "-"

    has_data = bool(
        numeric_or_zero(home_available) or numeric_or_zero(away_available)
    )
    if not has_data:
        return "-"

    home_confirmed = numeric_or_zero(row.get("Home_LineupConfirmed"))
    away_confirmed = numeric_or_zero(row.get("Away_LineupConfirmed"))
    if home_confirmed and away_confirmed:
        return "Confirmada"
    return "Provavel"


def render_upcoming_table(data: pd.DataFrame) -> None:
    """Mostra tabela principal de palpites futuros."""
    if data.empty:
        st.info("Nenhum palpite encontrado com os filtros atuais.")
        return

    st.subheader("Palpites futuros")
    col1, col2, col3 = st.columns((1.3, 1.0, 0.8))
    with col1:
        sort_mode = st.segmented_control(
            "Ordenar palpites",
            options=[
                "+EV primeiro",
                "Proximos",
                "Maior edge",
                "Maior prob.",
                "Melhor odd",
            ],
            default="+EV primeiro",
            key="upcoming_table_sort",
            width="stretch",
        )
    with col2:
        decision = st.pills(
            "Decisao",
            options=["Todos", "+EV", "Sem +EV"],
            default="Todos",
            key="upcoming_table_decision",
            width="stretch",
        )
    with col3:
        row_limit = st.selectbox(
            "Linhas",
            options=[25, 50, 100, 250],
            index=1,
            key="upcoming_table_rows",
        )
    query = st.text_input(
        "Buscar time, liga ou mercado",
        value="",
        key="upcoming_table_search",
    )

    table = data.copy()
    table["Jogo"] = table.apply(match_name, axis=1)
    table["Decisao"] = table["UiValueBet"].map({True: "+EV", False: "Passar"})
    table["Importancia"] = table["MatchImportance"].map(importance_label)
    table["Escalacao"] = table.apply(lineup_label, axis=1)
    table["RegraFiltro"] = table.apply(rule_filter_label, axis=1)
    table["ValueGap"] = table["ModelProb"] - table["ImpliedProb"]
    table["EdgeTable"] = table["Edge"]
    table["OddTable"] = table["BestOdd"]
    for optional_col in [
        "TeamStrength_Diff",
        "TeamStrength_Expected_Home",
        "ClubElo_DataAvailable",
    ]:
        if optional_col not in table.columns:
            table[optional_col] = np.nan
    if decision == "+EV":
        table = table[table["UiValueBet"]].copy()
    elif decision == "Sem +EV":
        table = table[~table["UiValueBet"]].copy()
    table = apply_text_search(
        table,
        query,
        ["LigaNome", "HomeTeam", "AwayTeam", "Market", "Selection", "Jogo"],
    )
    table = sort_table(table, sort_mode).head(row_limit)
    table = table[
        [
            "MatchDatetimeBR",
            "LigaNome",
            "Jogo",
            "Market",
            "Selection",
            "Decisao",
            "Importancia",
            "MatchImportance",
            "Escalacao",
            "LineupStrength_Diff",
            "MissingKeyPlayers_Diff",
            "xG_Expected_Total_Match_Roll5",
            "xG_Diff_Roll5",
            "Home_PreMatch_Rank",
            "Away_PreMatch_Rank",
            "RuleSource",
            "RegraFiltro",
            "TeamStrength_Diff",
            "TeamStrength_Expected_Home",
            "ClubElo_DataAvailable",
            "Elo_Diff",
            "Elo_Expected_Home",
            "BestBookmaker",
            "BestOdd",
            "ModelProb",
            "ImpliedProb",
            "ValueGap",
            "Edge",
            "RuleEvalROI",
            "RuleEvalBets",
        ]
    ]

    st.dataframe(
        style_dashboard_table(table),
        width="stretch",
        hide_index=True,
        column_config={
            "MatchDatetimeBR": st.column_config.DatetimeColumn("Data (BR)"),
            "LigaNome": "Liga",
            "Jogo": "Jogo",
            "Market": "Mercado",
            "Selection": "Selecao",
            "Decisao": "Decisao",
            "Importancia": "Importancia",
            "MatchImportance": st.column_config.ProgressColumn(
                "Imp.",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "Escalacao": "Escalacao",
            "LineupStrength_Diff": st.column_config.NumberColumn(
                "Forca XI",
                format="%.2f",
            ),
            "MissingKeyPlayers_Diff": st.column_config.NumberColumn(
                "Desf. chave",
                format="%.0f",
            ),
            "xG_Expected_Total_Match_Roll5": st.column_config.NumberColumn(
                "xG total",
                format="%.2f",
            ),
            "xG_Diff_Roll5": st.column_config.NumberColumn(
                "xG diff",
                format="%.2f",
            ),
            "Home_PreMatch_Rank": st.column_config.NumberColumn(
                "Rank casa",
                format="%.0f",
            ),
            "Away_PreMatch_Rank": st.column_config.NumberColumn(
                "Rank fora",
                format="%.0f",
            ),
            "RuleSource": "Regra",
            "RegraFiltro": "Filtro",
            "TeamStrength_Diff": st.column_config.NumberColumn(
                "Forca real",
                format="%.0f",
            ),
            "TeamStrength_Expected_Home": st.column_config.ProgressColumn(
                "Forca casa",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "ClubElo_DataAvailable": st.column_config.CheckboxColumn(
                "ClubElo",
                disabled=True,
            ),
            "Elo_Diff": st.column_config.NumberColumn("Elo diff", format="%.0f"),
            "Elo_Expected_Home": st.column_config.ProgressColumn(
                "Elo casa",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "BestBookmaker": "Casa usada",
            "BestOdd": st.column_config.NumberColumn("Odd usada", format="%.2f"),
            "ModelProb": st.column_config.ProgressColumn(
                "Prob. modelo",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "ImpliedProb": st.column_config.ProgressColumn(
                "Prob. implicita",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "ValueGap": st.column_config.NumberColumn(
                "Modelo - mercado",
                format="%.3f",
            ),
            "Edge": st.column_config.NumberColumn("Edge", format="%.3f"),
            "RuleEvalROI": st.column_config.NumberColumn(
                "ROI aval.",
                format="%.3f",
            ),
            "RuleEvalBets": st.column_config.NumberColumn(
                "Apostas aval.",
                format="%.0f",
            ),
        },
    )


def render_upcoming_charts(data: pd.DataFrame) -> None:
    """Renderiza graficos dos palpites futuros."""
    if data.empty:
        return

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            data,
            x="ModelProb",
            y="Edge",
            color="Market",
            symbol="UiValueBet",
            hover_data=[
                "LigaNome",
                "HomeTeam",
                "AwayTeam",
                "Selection",
                "RuleSource",
                "Elo_Diff",
                "MatchImportance",
                "LineupStrength_Diff",
                "MissingKeyPlayers_Diff",
                "xG_Expected_Total_Match_Roll5",
                "xG_Diff_Roll5",
                "BestBookmaker",
                "BestOdd",
            ],
            labels={
                "ModelProb": "Probabilidade do modelo",
                "Edge": "Edge",
                "MatchImportance": "Importancia",
                "LineupStrength_Diff": "Forca XI",
                "MissingKeyPlayers_Diff": "Desfalques-chave",
                "xG_Expected_Total_Match_Roll5": "xG total",
                "xG_Diff_Roll5": "xG diff",
                "Market": "Mercado",
            },
            title="Probabilidade vs. edge",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        summary = (
            data.groupby(["Market", "UiValueBet"], as_index=False)
            .size()
            .rename(columns={"size": "Palpites"})
        )
        fig = px.bar(
            summary,
            x="Market",
            y="Palpites",
            color="UiValueBet",
            labels={"Market": "Mercado", "UiValueBet": "+EV"},
            title="Volume por mercado",
        )
        st.plotly_chart(fig, width="stretch")


def render_bookmaker_odds(data: pd.DataFrame, odds: pd.DataFrame) -> None:
    """Mostra odds por casa para um jogo selecionado."""
    if data.empty or odds.empty:
        return

    st.subheader("Odds por casa")
    requested_bookmakers = sorted(
        {
            str(value)
            for value in data.get("RequestedBookmaker", pd.Series(dtype=str))
            .dropna()
            .unique()
            .tolist()
            if str(value).strip()
        }
    )
    if requested_bookmakers:
        st.caption(
            "Os palpites acima usam: "
            + ", ".join(requested_bookmakers)
            + ". A tabela abaixo mostra a comparacao completa da fonte."
        )
    fixtures = (
        data[["FixtureId", "MatchDatetimeBR", "LigaNome", "HomeTeam", "AwayTeam"]]
        .drop_duplicates()
        .sort_values("MatchDatetimeBR")
    )
    labels = {
        row["FixtureId"]: (
            f"{row['MatchDatetimeBR']:%d/%m %H:%M} BR | {row['LigaNome']} | "
            f"{row['HomeTeam']} x {row['AwayTeam']}"
        )
        for _, row in fixtures.iterrows()
    }
    selected_fixture = st.selectbox(
        "Jogo",
        options=fixtures["FixtureId"].tolist(),
        format_func=labels.get,
    )

    selected_odds = odds[odds["FixtureId"].eq(selected_fixture)].copy()
    if selected_odds.empty:
        st.info("Nao ha odds por casa para este jogo.")
        return

    pivot = selected_odds.pivot_table(
        index=["Market", "Selection"],
        columns="Bookmaker",
        values="Odd",
        aggfunc="max",
    ).reset_index()
    bookmaker_columns = [
        column for column in pivot.columns if column not in ["Market", "Selection"]
    ]
    for column in bookmaker_columns:
        pivot[column] = pd.to_numeric(pivot[column], errors="coerce")

    pivot["MelhorOdd"] = pivot[bookmaker_columns].max(axis=1)
    pivot["MelhorCasa"] = pivot[bookmaker_columns].idxmax(axis=1)
    pivot = pivot[
        ["Market", "Selection", "MelhorCasa", "MelhorOdd", *bookmaker_columns]
    ].sort_values(["Market", "Selection"])
    column_config = {
        "Market": "Mercado",
        "Selection": "Selecao",
        "MelhorCasa": "Melhor casa",
        "MelhorOdd": st.column_config.NumberColumn("Melhor odd", format="%.2f"),
    }
    column_config.update(
        {
            column: st.column_config.NumberColumn(column, format="%.2f")
            for column in bookmaker_columns
        }
    )
    st.dataframe(
        pivot,
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )


def load_model_comparison() -> pd.DataFrame:
    """Carrega o resumo de comparacao com odds vs sem odds."""
    if not COMPARISON_PATH.exists():
        return pd.DataFrame()
    return load_optional_csv(COMPARISON_PATH, COMPARISON_PATH.stat().st_mtime)


def build_comparison_delta(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula diferencas entre variante com odds e sem odds por mercado."""
    rows = []
    for market, market_data in data.groupby("Market"):
        variants = market_data.set_index("ModelVariant")
        if "Com odds" not in variants.index or "Sem odds" not in variants.index:
            continue
        with_odds = variants.loc["Com odds"]
        without_odds = variants.loc["Sem odds"]
        rows.append(
            {
                "Market": market,
                "DeltaAccuracy": with_odds["Accuracy"] - without_odds["Accuracy"],
                "DeltaLogLoss": with_odds["LogLoss"] - without_odds["LogLoss"],
                "DeltaBrierScore": (
                    with_odds.get("BrierScore", np.nan)
                    - without_odds.get("BrierScore", np.nan)
                ),
                "DeltaCalibrationECE": (
                    with_odds.get("CalibrationECE", np.nan)
                    - without_odds.get("CalibrationECE", np.nan)
                ),
                "DeltaROI": with_odds["ROI"] - without_odds["ROI"],
                "DeltaProfit": with_odds["TotalProfit"]
                - without_odds["TotalProfit"],
                "DeltaBets": with_odds["Bets"] - without_odds["Bets"],
            }
        )
    return pd.DataFrame(rows)


def render_model_comparison_page() -> None:
    """Renderiza comparacao dos modelos com odds e sem odds."""
    st.title("Apostasbot | Comparacao de modelos")
    st.caption("Modelo puro de futebol contra modelo com sinal de mercado.")
    render_pipeline_runner()

    data = load_model_comparison()
    if data.empty:
        st.info("Rode o pipeline com comparacao ativada para gerar o resumo.")
        return

    st.sidebar.header("Filtros")
    markets = sorted(data["Market"].dropna().unique().tolist())
    selected_markets = st.sidebar.multiselect(
        "Mercados",
        options=markets,
        default=markets,
    )
    filtered = data[data["Market"].isin(selected_markets)].copy()

    avg_aggs = {
        "Accuracy": ("Accuracy", "mean"),
        "LogLoss": ("LogLoss", "mean"),
        "ROI": ("ROI", "mean"),
        "TotalProfit": ("TotalProfit", "sum"),
    }
    if "BrierScore" in filtered.columns:
        avg_aggs["BrierScore"] = ("BrierScore", "mean")
    if "CalibrationECE" in filtered.columns:
        avg_aggs["CalibrationECE"] = ("CalibrationECE", "mean")
    avg_rows = filtered.groupby("ModelVariant", as_index=False).agg(**avg_aggs)
    with_odds = avg_rows[avg_rows["ModelVariant"].eq("Com odds")]
    without_odds = avg_rows[avg_rows["ModelVariant"].eq("Sem odds")]

    col1, col2, col3, col4 = st.columns(4)
    if not with_odds.empty and not without_odds.empty:
        odds_row = with_odds.iloc[0]
        no_odds_row = without_odds.iloc[0]
        col1.metric(
            "Delta acuracia",
            format_pct(odds_row["Accuracy"] - no_odds_row["Accuracy"]),
        )
        col2.metric(
            "Delta LogLoss",
            f"{odds_row['LogLoss'] - no_odds_row['LogLoss']:.4f}",
        )
        col3.metric(
            "Delta ROI",
            format_pct(odds_row["ROI"] - no_odds_row["ROI"]),
        )
        col4.metric(
            "Delta lucro",
            format_money(odds_row["TotalProfit"] - no_odds_row["TotalProfit"]),
        )

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            filtered,
            x="Market",
            y="ROI",
            color="ModelVariant",
            barmode="group",
            labels={"Market": "Mercado", "ROI": "ROI"},
            title="ROI por variante",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        metric_option = st.segmented_control(
            "Metrica de probabilidade",
            options=[
                metric
                for metric in ["LogLoss", "BrierScore", "CalibrationECE"]
                if metric in filtered.columns
            ],
            default="LogLoss",
            key="comparison_probability_metric",
            width="stretch",
        )
        metric_label = {
            "LogLoss": "LogLoss",
            "BrierScore": "Brier Score",
            "CalibrationECE": "ECE",
        }.get(metric_option, metric_option)
        fig = px.bar(
            filtered,
            x="Market",
            y=metric_option,
            color="ModelVariant",
            barmode="group",
            labels={"Market": "Mercado", metric_option: metric_label},
            title=f"{metric_label} por variante",
        )
        st.plotly_chart(fig, width="stretch")

    delta = build_comparison_delta(filtered)
    if not delta.empty:
        st.subheader("Diferenca com odds menos sem odds")
        st.dataframe(
            style_dashboard_table(delta),
            width="stretch",
            hide_index=True,
            column_config={
                "Market": "Mercado",
                "DeltaAccuracy": st.column_config.NumberColumn(
                    "Delta acuracia",
                    format="%.3f",
                ),
                "DeltaLogLoss": st.column_config.NumberColumn(
                    "Delta LogLoss",
                    format="%.4f",
                ),
                "DeltaBrierScore": st.column_config.NumberColumn(
                    "Delta Brier",
                    format="%.4f",
                ),
                "DeltaCalibrationECE": st.column_config.NumberColumn(
                    "Delta ECE",
                    format="%.3f",
                ),
                "DeltaROI": st.column_config.NumberColumn(
                    "Delta ROI",
                    format="%.3f",
                ),
                "DeltaProfit": st.column_config.NumberColumn(
                    "Delta lucro",
                    format="R$ %.2f",
                ),
                "DeltaBets": st.column_config.NumberColumn(
                    "Delta apostas",
                    format="%.0f",
                ),
            },
        )

    table_columns = [
        "Market",
        "ModelVariant",
        "FeatureCount",
        "TrainRows",
        "TestRows",
        "Accuracy",
        "LogLoss",
        "BrierScore",
        "CalibrationECE",
        "Bets",
        "ROI",
        "HitRate",
        "TotalProfit",
        "AvgEdge",
        "AvgOdd",
    ]
    table_columns = [column for column in table_columns if column in filtered.columns]
    table = filtered[table_columns].sort_values(["Market", "ModelVariant"])
    st.subheader("Resumo completo")
    st.dataframe(
        style_dashboard_table(table),
        width="stretch",
        hide_index=True,
        column_config={
            "Market": "Mercado",
            "ModelVariant": "Variante",
            "FeatureCount": "Features",
            "TrainRows": "Treino",
            "TestRows": "Teste",
            "Accuracy": st.column_config.NumberColumn("Acuracia", format="%.3f"),
            "LogLoss": st.column_config.NumberColumn("LogLoss", format="%.4f"),
            "BrierScore": st.column_config.NumberColumn("Brier", format="%.4f"),
            "CalibrationECE": st.column_config.NumberColumn("ECE", format="%.3f"),
            "Bets": st.column_config.NumberColumn("Apostas", format="%.0f"),
            "ROI": st.column_config.NumberColumn("ROI", format="%.3f"),
            "HitRate": st.column_config.NumberColumn("Acerto", format="%.3f"),
            "TotalProfit": st.column_config.NumberColumn(
                "Lucro",
                format="R$ %.2f",
            ),
            "AvgEdge": st.column_config.NumberColumn("Edge medio", format="%.3f"),
            "AvgOdd": st.column_config.NumberColumn("Odd media", format="%.2f"),
        },
    )


def load_filter_optimization() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega resumo e grade da otimizacao de filtros."""
    if FILTER_OPTIMIZATION_PATH.exists():
        summary = load_optional_csv(
            FILTER_OPTIMIZATION_PATH,
            FILTER_OPTIMIZATION_PATH.stat().st_mtime,
        )
    else:
        summary = pd.DataFrame()

    if FILTER_OPTIMIZATION_GRID_PATH.exists():
        grid = load_optional_csv(
            FILTER_OPTIMIZATION_GRID_PATH,
            FILTER_OPTIMIZATION_GRID_PATH.stat().st_mtime,
        )
    else:
        grid = pd.DataFrame()

    return summary, grid


def load_realistic_backtest() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Carrega artefatos do backtest realista."""
    paths = [
        REALISTIC_SUMMARY_PATH,
        REALISTIC_GRID_PATH,
        REALISTIC_BETS_PATH,
        REALISTIC_MONTHLY_PATH,
        REALISTIC_LEAGUE_PATH,
    ]
    frames = []
    for path in paths:
        if path.exists():
            frames.append(load_optional_csv(path, path.stat().st_mtime))
        else:
            frames.append(pd.DataFrame())
    return tuple(frames)


def load_ml_learning_artifacts() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Carrega artefatos de aprendizado, tuning e calibracao."""
    if FEATURE_IMPORTANCE_PATH.exists():
        feature_importance = load_optional_csv(
            FEATURE_IMPORTANCE_PATH,
            FEATURE_IMPORTANCE_PATH.stat().st_mtime,
        )
    else:
        feature_importance = pd.DataFrame()

    if XGB_TUNING_PATH.exists():
        tuning = load_optional_csv(
            XGB_TUNING_PATH,
            XGB_TUNING_PATH.stat().st_mtime,
        )
    else:
        tuning = pd.DataFrame()

    if PROBABILITY_BLEND_PATH.exists():
        probability_blend = load_optional_csv(
            PROBABILITY_BLEND_PATH,
            PROBABILITY_BLEND_PATH.stat().st_mtime,
        )
    else:
        probability_blend = pd.DataFrame()

    if CALIBRATION_CURVE_PATH.exists():
        calibration_curve = load_optional_csv(
            CALIBRATION_CURVE_PATH,
            CALIBRATION_CURVE_PATH.stat().st_mtime,
        )
    else:
        calibration_curve = pd.DataFrame()

    if CALIBRATION_METRICS_PATH.exists():
        calibration_metrics = load_optional_csv(
            CALIBRATION_METRICS_PATH,
            CALIBRATION_METRICS_PATH.stat().st_mtime,
        )
    else:
        calibration_metrics = pd.DataFrame()

    return (
        feature_importance,
        tuning,
        probability_blend,
        calibration_curve,
        calibration_metrics,
    )


FEATURE_FAMILY_COLORS = {
    "Odds/Mercado": "#2563eb",
    "xG": "#16a34a",
    "Forma/Gols": "#f59e0b",
    "Finalizacao": "#dc2626",
    "Escanteios": "#7c3aed",
    "Disciplina": "#b45309",
    "Forca real": "#0e7490",
    "Elo": "#0891b2",
    "Tabela": "#4f46e5",
    "Escalacao": "#be185d",
    "Arbitro": "#64748b",
    "Descanso": "#0f766e",
    "Outras": "#94a3b8",
}
FEATURE_LABEL_EXACT = {
    "Attack_Diff_Roll5": "Ataque: diferenca recente entre os times",
    "Away_Away_GF_Roll5": "Visitante fora: gols feitos recentes",
    "Away_BestVsSelectedGap": "Visitante: melhor odd comparada com a odd usada",
    "Away_GA_Roll5": "Visitante: gols sofridos recentes",
    "Away_GF_Roll5": "Visitante: gols feitos recentes",
    "Away_ImpliedMove": "Visitante: movimento da chance implicita",
    "Away_MaxAvgGap": "Visitante: melhor odd comparada com a media",
    "Away_OddsMove": "Visitante: movimento da odd",
    "Away_TeamStrength": "Forca real do visitante",
    "ClubElo_DataAvailable": "Tem leitura de forca real dos dois times",
    "Draw_BestVsSelectedGap": "Empate: melhor odd comparada com a odd usada",
    "Draw_MaxAvgGap": "Empate: melhor odd comparada com a media",
    "Draw_OddsMove": "Empate: movimento da odd",
    "Elo_Diff": "Elo interno: diferenca entre os times",
    "Elo_Expected_Home": "Elo interno: chance do mandante",
    "Expected_Total_Goals_Form_Roll5": "Total esperado de gols pela forma recente",
    "Home_BestVsSelectedGap": "Mandante: melhor odd comparada com a odd usada",
    "Home_Away_RestDays": "Mandante: descanso fora de casa",
    "Home_FailedToScore_Roll5": "Mandante: vezes recentes sem marcar",
    "Home_GA_Roll5": "Mandante: gols sofridos recentes",
    "Home_GF_Roll5": "Mandante: gols feitos recentes",
    "Home_Home_GF_Roll5": "Mandante em casa: gols feitos recentes",
    "Home_ImpliedMove": "Mandante: movimento da chance implicita",
    "Home_MaxAvgGap": "Mandante: melhor odd comparada com a media",
    "Home_OddsMove": "Mandante: movimento da odd",
    "Home_TeamStrength": "Forca real do mandante",
    "Home_xGFor_Roll5": "Mandante: xG recente",
    "MatchImportance": "Importancia do jogo",
    "AH_Line": "Linha asiatica do jogo",
    "AH_AbsLine": "Forca da linha asiatica",
    "AH_HomeFavored": "Mandante favorito na linha asiatica",
    "AH_AwayFavored": "Visitante favorito na linha asiatica",
    "AH_PickEm": "Jogo equilibrado na linha asiatica",
    "AH_HomeOdds": "Odd asiatica do mandante",
    "AH_AwayOdds": "Odd asiatica do visitante",
    "AH_Overround": "Margem da linha asiatica",
    "AH_NoVigHomeCover": "Chance limpa do mandante cobrir a linha",
    "AH_NoVigAwayCover": "Chance limpa do visitante cobrir a linha",
    "AH_FavoriteCoverProb": "Chance limpa do favorito cobrir a linha",
    "AH_UnderdogCoverProb": "Chance limpa do zebra cobrir a linha",
    "AH_HomeStrengthMove": "Movimento da linha a favor do mandante",
    "AH_AbsLineMove": "Mudanca na forca da linha asiatica",
    "AH_HomeOddsMove": "Movimento da odd asiatica do mandante",
    "AH_AwayOddsMove": "Movimento da odd asiatica do visitante",
    "League_TotalGoals_Pregame": "Media recente de gols da liga",
    "League_Over25Rate_Pregame": "Taxa recente de over 2.5 da liga",
    "League_HomeGoals_Pregame": "Media recente de gols do mandante na liga",
    "League_AwayGoals_Pregame": "Media recente de gols do visitante na liga",
    "League_FirstHalfGoals_Pregame": "Media recente de gols no 1o tempo da liga",
    "League_ShotsTotal_Pregame": "Media recente de finalizacoes da liga",
    "League_xGTotal_Pregame": "Media recente de xG da liga",
    "League_OddOver25_Pregame": "Odd media recente do over 2.5 na liga",
    "Home_Attack_vs_League": "Ataque do mandante contra a media da liga",
    "Away_Attack_vs_League": "Ataque do visitante contra a media da liga",
    "Expected_Total_vs_League": "Total esperado do jogo contra a media da liga",
    "Shots_Total_vs_League": "Finalizacoes do jogo contra a media da liga",
    "FirstHalfGoals_vs_League": "Gols de 1o tempo contra a media da liga",
    "xG_Total_vs_League": "xG esperado do jogo contra a media da liga",
    "NoVig_Prob_Away": "Chance do mercado no visitante",
    "NoVig_Prob_Draw": "Chance do mercado no empate",
    "NoVig_Prob_Home": "Chance do mercado no mandante",
    "NoVig_Prob_Over25": "Chance do mercado no over 2.5",
    "NoVig_Prob_Under25": "Chance do mercado no under 2.5",
    "Odd_Away": "Odd usada no visitante",
    "Odd_Draw": "Odd usada no empate",
    "Odd_Home": "Odd usada no mandante",
    "Odd_Over25": "Odd usada no over 2.5",
    "Odd_Under25": "Odd usada no under 2.5",
    "Odds_Quality_Result": "Qualidade da linha de resultado",
    "Odds_Quality_Total": "Qualidade da linha de gols",
    "Over25_MaxAvgGap": "Linha de gols: melhor odd vs media",
    "Points_Diff_Roll5": "Pontos recentes: diferenca entre os times",
    "Raw_Implied_Prob_Away": "Chance bruta da odd no visitante",
    "Raw_Implied_Prob_Draw": "Chance bruta da odd no empate",
    "Raw_Implied_Prob_Home": "Chance bruta da odd no mandante",
    "Raw_Implied_Prob_Over25": "Chance bruta da odd no over 2.5",
    "Raw_Implied_Prob_Under25": "Chance bruta da odd no under 2.5",
    "RestDays_Diff": "Descanso: diferenca entre os times",
    "Result_MaxAvgGap": "Linha de resultado: melhor odd vs media",
    "ShotsOnTarget_Total_Roll5": "Finalizacoes no alvo somadas",
    "TeamStrength_Diff": "Forca real: diferenca entre os times",
    "TeamStrength_Expected_Away": "Forca real: chance do visitante",
    "TeamStrength_Expected_Home": "Forca real: chance do mandante",
    "TopClash": "Duelo forte entre times do alto da tabela",
    "Total_Goals_Form_Roll5": "Total de gols recente dos dois times",
    "Totals_MaxAvgGap": "Linha de gols: dispersao entre casas",
    "Under25_MaxAvgGap": "Linha de under: melhor odd vs media",
    "Venue_Attack_Diff_Roll5": "Ataque em casa/fora: diferenca recente",
    "Venue_Total_Goals_Form_Roll5": "Total de gols em casa/fora",
    "xG_Expected_Total_Match_Roll5": "xG total esperado para o jogo",
    "xG_Total_Roll5": "xG recente somado dos dois times",
}
FEATURE_TOKEN_LABELS = {
    "AH": "linha asiatica",
    "Attack": "ataque",
    "Avg": "media",
    "Available": "disponivel",
    "Away": "visitante",
    "Best": "melhor odd",
    "Both": "ambos",
    "Cards": "cartoes",
    "Chance": "chance",
    "CleanSheet": "sem sofrer gol",
    "Closing": "fechamento",
    "ClubElo": "forca real",
    "ConversionRate": "taxa de conversao",
    "Corners": "escanteios",
    "Data": "dados",
    "Defense": "defesa",
    "Diff": "diferenca",
    "Discipline": "disciplina",
    "Draw": "empate",
    "Evening": "noite",
    "Expected": "esperado",
    "ExpectedHome": "chance do mandante",
    "FailedToScore": "ficou sem marcar",
    "Family": "grupo",
    "FirstHalf": "1o tempo",
    "Fouls": "faltas",
    "GA": "gols sofridos",
    "GF": "gols feitos",
    "GoalDiff": "saldo de gols",
    "Goals": "gols",
    "Half": "tempo",
    "Home": "mandante",
    "Importance": "importancia",
    "Implied": "implicita",
    "IsMidweek": "meio de semana",
    "IsWeekend": "fim de semana",
    "Gap": "distancia",
    "Kickoff": "horario",
    "League": "liga",
    "Lineup": "escalacao",
    "LongRest": "descanso longo",
    "LossRate": "taxa de derrota",
    "Match": "jogo",
    "Matches": "jogos",
    "MaxAvgGap": "melhor odd vs media",
    "Missing": "desfalques",
    "Move": "movimento",
    "No": "sem",
    "Odds": "odds",
    "Over25": "over 2.5",
    "Points": "pontos",
    "Pre": "antes do jogo",
    "PreMatch": "antes do jogo",
    "Prob": "chance",
    "Quality": "qualidade",
    "Raw": "bruta",
    "Rate": "taxa",
    "Real": "real",
    "RedCards": "cartoes vermelhos",
    "Referee": "arbitro",
    "RestDays": "dias de descanso",
    "Result": "resultado",
    "Roll5": "ultimos 5 jogos",
    "SeasonProgress": "andar da temporada",
    "SecondHalf": "2o tempo",
    "Sheet": "limpo",
    "ShortRest": "descanso curto",
    "ShotAccuracy": "precisao nas finalizacoes",
    "ShotPressure": "pressao de chute",
    "Shots": "finalizacoes",
    "SOTAllowedRate": "chutes no alvo cedidos",
    "Strength": "forca",
    "Selected": "usada",
    "Team": "time",
    "TeamStrength": "forca real",
    "TitlePressure": "pressao por titulo",
    "TopClash": "duelo forte",
    "Total": "total",
    "Under25": "under 2.5",
    "Venue": "casa/fora",
    "Vig": "vig",
    "Vs": "contra",
    "WinRate": "taxa de vitoria",
    "xG": "xG",
    "YellowCards": "cartoes amarelos",
}


def feature_family(feature: object) -> str:
    """Agrupa uma feature em uma familia legivel."""
    text = str(feature).lower()
    if text.startswith("ah_"):
        return "Linha asiatica"
    if text.startswith("league_") or text.endswith("_vs_league"):
        return "Contexto da liga"
    if any(token in text for token in ["odd", "implied", "novig", "overround"]):
        return "Odds/Mercado"
    if "xg" in text:
        return "xG"
    if any(token in text for token in ["teamstrength", "clubelo"]):
        return "Forca real"
    if "elo" in text:
        return "Elo"
    if any(token in text for token in ["referee", "arbitro"]):
        return "Arbitro"
    if any(token in text for token in ["lineup", "missing"]):
        return "Escalacao"
    if any(
        token in text
        for token in [
            "rank",
            "pointspergame",
            "titlepressure",
            "relegation",
            "importance",
            "topclash",
            "seasonprogress",
        ]
    ):
        return "Tabela"
    if any(token in text for token in ["restdays", "restdays_diff"]):
        return "Descanso"
    if any(token in text for token in ["shot", "conversion", "sot"]):
        return "Finalizacao"
    if "corner" in text:
        return "Escanteios"
    if any(token in text for token in ["card", "foul", "discipline"]):
        return "Disciplina"
    if any(
        token in text
        for token in ["goal", "over25", "attack", "defense", "winrate", "lossrate"]
    ):
        return "Forma/Gols"
    return "Outras"


def feature_readable_name(feature: object) -> str:
    """Transforma o nome tecnico da feature em leitura humana."""
    raw = str(feature).strip()
    if raw in FEATURE_LABEL_EXACT:
        return FEATURE_LABEL_EXACT[raw]

    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw)
    normalized = normalized.replace("Over25", "Over25").replace("Under25", "Under25")
    tokens = [token for token in normalized.split("_") if token]

    translated: list[str] = []
    for token in tokens:
        translated.append(FEATURE_TOKEN_LABELS.get(token, token.lower()))

    text = " - ".join(part for part in translated if part)
    text = text.replace(" - ultimos 5 jogos", " (ultimos 5 jogos)")
    text = text.replace(" - antes do jogo", " antes do jogo")
    text = re.sub(r"\s+", " ", text).strip(" -")
    if not text:
        return raw
    return text[:1].upper() + text[1:]


def learning_quality_label(brier_score: float, ece: float) -> str:
    """Classifica a confianca geral das probabilidades."""
    if pd.isna(brier_score) or pd.isna(ece):
        return "Sem leitura"
    if brier_score <= 0.20 and ece <= 0.03:
        return "Muito boa"
    if brier_score <= 0.24 and ece <= 0.05:
        return "Boa"
    if brier_score <= 0.28 and ece <= 0.08:
        return "Regular"
    return "Pede ajuste"


def enrich_feature_importance(data: pd.DataFrame) -> pd.DataFrame:
    """Prepara importancia de features para visualizacao."""
    if data.empty:
        return data

    enriched = data.copy()
    enriched["Family"] = enriched["Feature"].map(feature_family)
    enriched["FeatureLabel"] = enriched["Feature"].map(feature_readable_name)
    enriched["Importance"] = pd.to_numeric(
        enriched["Importance"],
        errors="coerce",
    ).fillna(0.0)
    enriched["ImportanceShare"] = pd.to_numeric(
        enriched.get("ImportanceShare", 0.0),
        errors="coerce",
    ).fillna(0.0)
    enriched["Rank"] = pd.to_numeric(enriched["Rank"], errors="coerce")
    enriched = enriched.sort_values(
        ["Market", "Importance"],
        ascending=[True, False],
        kind="mergesort",
    )
    enriched["CumulativeShare"] = enriched.groupby("Market")[
        "ImportanceShare"
    ].cumsum()
    return enriched


def render_learning_overview(
    feature_importance: pd.DataFrame,
    tuning: pd.DataFrame,
    probability_blend: pd.DataFrame,
    calibration_metrics: pd.DataFrame,
) -> None:
    """Mostra um resumo visual e facil de ler sobre o aprendizado."""
    markets = 0
    feature_count = 0
    top_family = "-"
    top_family_share = np.nan
    top_signal = "-"
    if not feature_importance.empty:
        markets = feature_importance["Market"].dropna().nunique()
        feature_count = feature_importance["Feature"].dropna().nunique()
        family_summary = feature_importance.groupby("Family", as_index=False).agg(
            ImportanceShare=("ImportanceShare", "sum")
        )
        family_summary = family_summary.sort_values(
            "ImportanceShare",
            ascending=False,
            kind="mergesort",
        )
        top_family = str(family_summary.iloc[0]["Family"])
        top_family_share = float(family_summary.iloc[0]["ImportanceShare"])
        top_feature_row = feature_importance.sort_values(
            "Importance",
            ascending=False,
            kind="mergesort",
        ).iloc[0]
        top_signal = str(top_feature_row.get("FeatureLabel", top_feature_row["Feature"]))

    best_logloss = np.nan
    if not tuning.empty and "ValidationLogLoss" in tuning.columns:
        best_logloss = float(
            pd.to_numeric(
                tuning["ValidationLogLoss"],
                errors="coerce",
            ).min()
        )

    weighted_brier = np.nan
    weighted_ece = np.nan
    if not calibration_metrics.empty:
        weighted_brier = float(
            np.average(
                calibration_metrics["BrierScore"],
                weights=calibration_metrics["Rows"],
            )
        )
        weighted_ece = float(
            np.average(
                calibration_metrics["ECE"],
                weights=calibration_metrics["Rows"],
            )
        )

    blend_gain = np.nan
    blend_alpha = np.nan
    if not probability_blend.empty:
        blend = enrich_probability_blend(probability_blend)
        best_rows = blend[blend["IsBest"]].copy()
        if best_rows.empty:
            best_rows = (
                blend.sort_values(
                    ["Market", "ValidationLogLoss"],
                    ascending=[True, True],
                    kind="mergesort",
                )
                .groupby("Market", as_index=False)
                .head(1)
            )
        if not best_rows.empty:
            blend_gain = float(best_rows["ImprovementVsModel"].mean())
            blend_alpha = float(best_rows["AlphaModel"].mean())

    st.subheader("Leitura rapida")
    st.caption(
        "Aqui a ideia e responder tres perguntas sem jargao: no que o modelo "
        "mais olha, se as chances que ele calcula fazem sentido e quanto o "
        "mercado ajuda."
    )
    card_cols = st.columns(4)

    with card_cols[0]:
        with st.container(border=True):
            st.metric("Mercados e sinais", f"{markets:.0f} mercados")
            st.caption(
                f"{feature_count:.0f} sinais diferentes acompanhados."
            )

    with card_cols[1]:
        with st.container(border=True):
            st.metric(
                "O que mais pesa",
                top_family,
                delta=None if pd.isna(top_family_share) else format_pct(top_family_share),
            )
            st.caption(f"Sinal mais forte: {top_signal}")

    with card_cols[2]:
        with st.container(border=True):
            st.metric(
                "Confianca das chances",
                learning_quality_label(weighted_brier, weighted_ece),
            )
            chance_text = (
                "Sem leitura"
                if pd.isna(weighted_brier)
                else f"Precisao {weighted_brier:.3f} | equilibrio {weighted_ece:.1%}"
            )
            st.caption(chance_text)

    with card_cols[3]:
        with st.container(border=True):
            blend_title = "Mercado ajuda?"
            blend_value = (
                "Sem leitura"
                if pd.isna(blend_gain)
                else ("Ajuda" if blend_gain > 0 else "Quase nada")
            )
            st.metric(blend_title, blend_value)
            blend_text = (
                "Sem comparacao suficiente."
                if pd.isna(blend_gain)
                else (
                    f"Peso medio do modelo: {format_pct(blend_alpha)} | "
                    f"ganho medio: {blend_gain:.4f}"
                )
            )
            st.caption(blend_text)

    if not pd.isna(best_logloss):
        st.caption(
            "Menor erro visto nos testes automáticos: "
            f"{best_logloss:.4f}. Menor e melhor."
        )


def render_feature_learning(feature_importance: pd.DataFrame) -> None:
    """Renderiza visualizacoes de importancia de features."""
    if feature_importance.empty:
        st.info("Importancia de features ainda nao foi gerada.")
        return

    st.caption(
        "Estas barras mostram quais sinais mais mexem na decisao do modelo. "
        "Quanto maior a barra, mais o modelo olha para esse ponto."
    )

    markets = sorted(feature_importance["Market"].dropna().unique().tolist())
    left, right, spacer = st.columns((1.5, 1.0, 1.2))
    with left:
        selected_market = st.segmented_control(
            "Mercado",
            options=markets,
            default=markets[0],
            key="ml_feature_market",
            width="stretch",
        )
    with right:
        top_n = st.selectbox(
            "Quantidade",
            options=[10, 15, 20, 30],
            index=1,
            key="ml_feature_top_n",
        )
    with spacer:
        family_filter = st.multiselect(
            "Familias",
            options=list(FEATURE_FAMILY_COLORS),
            default=[],
            key="ml_feature_family_filter",
            placeholder="Todas",
        )

    market_features = feature_importance[
        feature_importance["Market"].eq(selected_market)
    ].copy()
    if family_filter:
        market_features = market_features[
            market_features["Family"].isin(family_filter)
        ].copy()
    market_features = market_features.sort_values(
        "Importance",
        ascending=False,
        kind="mergesort",
    )

    if market_features.empty:
        st.info("Nenhuma feature encontrada para os filtros atuais.")
        return

    top_features = market_features.head(int(top_n)).copy()
    family_summary = (
        market_features.groupby("Family", as_index=False)
        .agg(
            ImportanceShare=("ImportanceShare", "sum"),
            Importance=("Importance", "sum"),
            Features=("Feature", "count"),
        )
        .sort_values("ImportanceShare", ascending=False, kind="mergesort")
    )

    chart_col, family_col = st.columns((1.8, 1.0))
    with chart_col:
        fig = px.bar(
            top_features.sort_values("Importance", ascending=True),
            x="Importance",
            y="FeatureLabel",
            orientation="h",
            color="Family",
            color_discrete_map=FEATURE_FAMILY_COLORS,
            hover_data={
                "Feature": True,
                "ImportanceShare": ":.2%",
                "CumulativeShare": ":.2%",
                "Rank": ":.0f",
            },
            labels={
                "Importance": "Peso",
                "FeatureLabel": "Sinal",
                "Family": "Grupo",
                "ImportanceShare": "Participacao",
                "CumulativeShare": "Acumulado",
            },
            title="Sinais que mais pesam",
        )
        fig.update_layout(
            height=max(420, 26 * len(top_features)),
            margin=dict(l=10, r=10, t=55, b=10),
            legend_title_text="",
        )
        st.plotly_chart(fig, width="stretch")

    with family_col:
        fig = px.bar(
            family_summary.sort_values("ImportanceShare", ascending=True),
            x="ImportanceShare",
            y="Family",
            orientation="h",
            color="Family",
            color_discrete_map=FEATURE_FAMILY_COLORS,
            text=family_summary.sort_values(
                "ImportanceShare",
                ascending=True,
            )["ImportanceShare"].map(lambda value: f"{value:.1%}"),
            labels={
                "ImportanceShare": "Participacao",
                "Family": "Familia",
            },
            title="Grupos que mais pesam",
        )
        fig.update_layout(
            height=420,
            showlegend=False,
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    st.dataframe(
        style_dashboard_table(
            top_features[
                [
                    "Rank",
                    "FeatureLabel",
                    "Family",
                    "Importance",
                    "ImportanceShare",
                    "CumulativeShare",
                ]
            ]
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("#", format="%.0f"),
            "FeatureLabel": "Sinal",
            "Family": "Grupo",
            "Importance": st.column_config.ProgressColumn(
                "Peso",
                format="%.4f",
                min_value=0.0,
                max_value=float(top_features["Importance"].max()),
            ),
            "ImportanceShare": st.column_config.NumberColumn(
                "Participacao",
                format="%.2%",
            ),
            "CumulativeShare": st.column_config.NumberColumn(
                "Acumulado",
                format="%.2%",
            ),
        },
    )


def render_tuning_learning(tuning: pd.DataFrame) -> None:
    """Renderiza leitura visual do tuning temporal."""
    if tuning.empty:
        st.info("Tuning XGBoost ainda nao foi gerado.")
        return

    st.caption(
        "Aqui o app testa varias regulagens do motor e guarda a que erra menos "
        "fora da amostra. Menor erro e melhor."
    )

    tuning_table = tuning.copy()
    tuning_table["ValidationLogLoss"] = pd.to_numeric(
        tuning_table["ValidationLogLoss"],
        errors="coerce",
    )
    tuning_table = tuning_table.sort_values(
        ["Market", "ValidationLogLoss"],
        ascending=[True, True],
        kind="mergesort",
    )
    best_rows = tuning_table.groupby("Market", as_index=False).first()

    col1, col2 = st.columns((1.6, 1.0))
    with col1:
        fig = px.line(
            tuning_table.sort_values(["Market", "Trial"], kind="mergesort"),
            x="Trial",
            y="ValidationLogLoss",
            color="Market",
            markers=True,
            labels={
                "Trial": "Tentativa",
                "ValidationLogLoss": "Erro fora da amostra",
                "Market": "Mercado",
            },
            title="Como cada tentativa se saiu",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.bar(
            best_rows.sort_values("ValidationLogLoss", ascending=False),
            x="ValidationLogLoss",
            y="Market",
            orientation="h",
            color="Market",
            labels={
                "ValidationLogLoss": "Menor erro",
                "Market": "Mercado",
            },
            title="Melhor regulagem por mercado",
        )
        fig.update_layout(
            height=420,
            showlegend=False,
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    st.dataframe(
        style_dashboard_table(
            best_rows[
                [
                    "Market",
                    "Trial",
                    "ValidationLogLoss",
                    "n_estimators",
                    "learning_rate",
                    "max_depth",
                    "subsample",
                    "colsample_bytree",
                    "min_child_weight",
                    "reg_lambda",
                    "reg_alpha",
                ]
            ]
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Market": "Mercado",
            "Trial": st.column_config.NumberColumn("Tentativa", format="%.0f"),
            "ValidationLogLoss": st.column_config.NumberColumn(
                "Erro final",
                format="%.4f",
            ),
            "n_estimators": st.column_config.NumberColumn(
                "Qtd. arvores",
                format="%.0f",
            ),
            "learning_rate": st.column_config.NumberColumn(
                "Passo",
                format="%.3f",
            ),
            "max_depth": st.column_config.NumberColumn(
                "Profundidade",
                format="%.0f",
            ),
            "subsample": st.column_config.NumberColumn("Amostra", format="%.2f"),
            "colsample_bytree": st.column_config.NumberColumn(
                "Colunas",
                format="%.2f",
            ),
            "min_child_weight": st.column_config.NumberColumn(
                "Min. base",
                format="%.1f",
            ),
            "reg_lambda": st.column_config.NumberColumn("Freio L2", format="%.2f"),
            "reg_alpha": st.column_config.NumberColumn("Freio L1", format="%.2f"),
        },
    )

    with st.expander("Ver todas as regulagens"):
        st.dataframe(tuning_table, width="stretch", hide_index=True)


def enrich_calibration_curve(data: pd.DataFrame) -> pd.DataFrame:
    """Normaliza colunas numericas da curva de calibracao."""
    if data.empty:
        return data

    enriched = data.copy()
    numeric_cols = [
        "Bin",
        "BinStart",
        "BinEnd",
        "MeanPredictedProb",
        "ObservedRate",
        "Count",
        "CalibrationGap",
        "AbsCalibrationGap",
        "BrierScore",
    ]
    for column in numeric_cols:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    enriched["Probabilidade"] = enriched["MeanPredictedProb"]
    enriched["FrequenciaReal"] = enriched["ObservedRate"]
    enriched["Faixa"] = (
        enriched["BinStart"].map(lambda value: f"{value:.0%}")
        + " a "
        + enriched["BinEnd"].map(lambda value: f"{value:.0%}")
    )
    return enriched


def enrich_calibration_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Normaliza metricas agregadas de calibracao."""
    if data.empty:
        return data

    enriched = data.copy()
    numeric_cols = [
        "Rows",
        "MeanPredictedProb",
        "ObservedRate",
        "BrierScore",
        "ECE",
        "MaxAbsGap",
        "CalibrationBias",
    ]
    for column in numeric_cols:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    return enriched


def render_calibration_learning(
    calibration_curve: pd.DataFrame,
    calibration_metrics: pd.DataFrame,
) -> None:
    """Renderiza Brier Score, ECE e curva prevista vs. observada."""
    if calibration_curve.empty and calibration_metrics.empty:
        st.info("Rode o pipeline novamente para gerar curvas de calibracao.")
        return

    st.caption(
        "A pergunta aqui e simples: quando o modelo diz 60%, isso vira algo "
        "perto de 60% na vida real?"
    )

    curve = enrich_calibration_curve(calibration_curve)
    metrics = enrich_calibration_metrics(calibration_metrics)
    markets = sorted(
        set(curve.get("Market", pd.Series(dtype=str)).dropna().unique()).union(
            set(metrics.get("Market", pd.Series(dtype=str)).dropna().unique())
        )
    )
    if not markets:
        st.info("Nao ha mercados calibrados nos artefatos atuais.")
        return

    col_market, col_outcomes = st.columns((1.2, 1.8))
    with col_market:
        selected_market = st.segmented_control(
            "Mercado",
            options=markets,
            default=markets[0],
            key="ml_calibration_market",
            width="stretch",
        )

    market_curve = curve[curve["Market"].eq(selected_market)].copy()
    market_metrics = metrics[metrics["Market"].eq(selected_market)].copy()
    outcomes = sorted(
        set(market_curve.get("Outcome", pd.Series(dtype=str)).dropna().unique()).union(
            set(market_metrics.get("Outcome", pd.Series(dtype=str)).dropna().unique())
        )
    )
    with col_outcomes:
        selected_outcomes = st.multiselect(
            "Eventos",
            options=outcomes,
            default=outcomes,
            key="ml_calibration_outcomes",
        )

    if selected_outcomes:
        market_curve = market_curve[
            market_curve["Outcome"].isin(selected_outcomes)
        ].copy()
        market_metrics = market_metrics[
            market_metrics["Outcome"].isin(selected_outcomes)
        ].copy()

    if market_metrics.empty:
        st.info("Nao ha metricas para os filtros atuais.")
        return

    total_rows = float(market_metrics["Rows"].sum())
    weighted_brier = np.average(
        market_metrics["BrierScore"],
        weights=market_metrics["Rows"],
    )
    weighted_ece = np.average(
        market_metrics["ECE"],
        weights=market_metrics["Rows"],
    )
    mean_bias = np.average(
        market_metrics["CalibrationBias"],
        weights=market_metrics["Rows"],
    )
    max_gap = float(market_metrics["MaxAbsGap"].max())

    metric_cols = st.columns(4)
    metric_cols[0].metric("Precisao das chances", f"{weighted_brier:.4f}")
    metric_cols[1].metric("Equilibrio", format_pct(weighted_ece))
    metric_cols[2].metric("Tendencia media", format_pct(mean_bias))
    metric_cols[3].metric("Maior desvio", format_pct(max_gap))
    st.caption(f"Amostra avaliada: {total_rows:,.0f} probabilidades por evento.")

    if market_curve.empty:
        st.info("Nao ha pontos de curva para os filtros atuais.")
        return

    chart_col, gap_col = st.columns((1.35, 1.0))
    with chart_col:
        fig = px.line(
            market_curve.sort_values(
                ["Outcome", "MeanPredictedProb"],
                kind="mergesort",
            ),
            x="Probabilidade",
            y="FrequenciaReal",
            color="Outcome",
            markers=True,
            hover_data={
                "Faixa": True,
                "Count": ":.0f",
                "BrierScore": ":.4f",
                "CalibrationGap": ":.2%",
            },
            labels={
                "Probabilidade": "Chance dita pelo modelo",
                "FrequenciaReal": "O que aconteceu",
                "Outcome": "Evento",
            },
            title="Chance dita x chance real",
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Calibracao perfeita",
                line=dict(color="#94a3b8", dash="dash"),
            )
        )
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, width="stretch")

    with gap_col:
        fig = px.bar(
            market_curve,
            x="Faixa",
            y="CalibrationGap",
            color="Outcome",
            hover_data={"Count": ":.0f", "AbsCalibrationGap": ":.2%"},
            labels={
                "Faixa": "Faixa de chance",
                "CalibrationGap": "Real - previsto",
                "Outcome": "Evento",
            },
            title="Desvio por faixa",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Resumo por evento")
    st.dataframe(
        style_dashboard_table(
            market_metrics[
                [
                    "Outcome",
                    "Rows",
                    "MeanPredictedProb",
                    "ObservedRate",
                    "BrierScore",
                    "ECE",
                    "MaxAbsGap",
                    "CalibrationBias",
                ]
            ]
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Outcome": "Evento",
            "Rows": st.column_config.NumberColumn("Amostras", format="%.0f"),
            "MeanPredictedProb": st.column_config.NumberColumn(
                "Chance media dita",
                format="%.2%",
            ),
            "ObservedRate": st.column_config.NumberColumn(
                "O que aconteceu",
                format="%.2%",
            ),
            "BrierScore": st.column_config.NumberColumn("Precisao", format="%.4f"),
            "ECE": st.column_config.NumberColumn("Equilibrio", format="%.2%"),
            "MaxAbsGap": st.column_config.NumberColumn("Maior desvio", format="%.2%"),
            "CalibrationBias": st.column_config.NumberColumn(
                "Tendencia",
                format="%.2%",
            ),
        },
    )


def enrich_probability_blend(data: pd.DataFrame) -> pd.DataFrame:
    """Normaliza diagnostico de blend entre modelo e mercado."""
    if data.empty:
        return data

    enriched = data.copy()
    numeric_cols = [
        "AlphaModel",
        "AlphaMarket",
        "ValidationLogLoss",
        "ModelLogLoss",
        "MarketLogLoss",
        "ImprovementVsModel",
        "ImprovementVsMarket",
    ]
    for column in numeric_cols:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    if "IsBest" in enriched.columns:
        enriched["IsBest"] = enriched["IsBest"].astype(str).str.lower().isin(
            ["true", "1", "sim"]
        )
    return enriched


def render_probability_blend_learning(probability_blend: pd.DataFrame) -> None:
    """Mostra quanto o modelo deve pesar contra a probabilidade de mercado."""
    if probability_blend.empty:
        st.info("Rode o pipeline novamente para gerar o blend modelo x mercado.")
        return

    st.caption(
        "Aqui a gente mede quando vale confiar mais no modelo puro e quando "
        "vale ouvir mais o mercado."
    )

    blend = enrich_probability_blend(probability_blend)
    markets = sorted(blend["Market"].dropna().unique().tolist())
    selected_markets = st.multiselect(
        "Mercados",
        options=markets,
        default=markets,
        key="ml_blend_markets",
    )
    filtered = blend[blend["Market"].isin(selected_markets)].copy()
    if filtered.empty:
        st.info("Nenhum mercado selecionado.")
        return

    best = filtered[filtered["IsBest"]].copy()
    if best.empty:
        best = (
            filtered.sort_values(
                ["Market", "ValidationLogLoss"],
                kind="mergesort",
            )
            .groupby("Market", as_index=False)
            .head(1)
        )

    avg_alpha = float(best["AlphaModel"].mean()) if not best.empty else np.nan
    avg_gain = (
        float(best["ImprovementVsModel"].mean()) if not best.empty else np.nan
    )
    cols = st.columns(4)
    cols[0].metric("Mercados com blend", f"{best['Market'].nunique():.0f}")
    cols[1].metric(
        "Quanto o modelo pesa",
        "-" if pd.isna(avg_alpha) else format_pct(avg_alpha),
    )
    cols[2].metric(
        "Ganho medio",
        "-" if pd.isna(avg_gain) else f"{avg_gain:.4f}",
    )
    cols[3].metric(
        "Menor erro final",
        f"{best['ValidationLogLoss'].min():.4f}",
    )

    left, right = st.columns((1.5, 1.0))
    with left:
        fig = px.line(
            filtered.sort_values(["Market", "AlphaModel"], kind="mergesort"),
            x="AlphaModel",
            y="ValidationLogLoss",
            color="Market",
            markers=True,
            hover_data={
                "AlphaMarket": ":.2f",
                "ImprovementVsModel": ":.4f",
                "ImprovementVsMarket": ":.4f",
            },
            labels={
                "AlphaModel": "Peso dado ao modelo",
                "ValidationLogLoss": "Erro fora da amostra",
                "Market": "Mercado",
            },
            title="Quanto ouvir o modelo e o mercado",
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, width="stretch")

    with right:
        weight_plot = best.melt(
            id_vars=["Market"],
            value_vars=["AlphaModel", "AlphaMarket"],
            var_name="Fonte",
            value_name="Peso",
        )
        weight_plot["Fonte"] = weight_plot["Fonte"].map(
            {"AlphaModel": "Modelo", "AlphaMarket": "Mercado"}
        )
        fig = px.bar(
            weight_plot,
            x="Market",
            y="Peso",
            color="Fonte",
            barmode="stack",
            labels={"Market": "Mercado", "Peso": "Peso"},
            title="Peso final escolhido",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Leitura por mercado")
    st.dataframe(
        style_dashboard_table(
            best[
                [
                    "Market",
                    "AlphaModel",
                    "AlphaMarket",
                    "ValidationLogLoss",
                    "ModelLogLoss",
                    "MarketLogLoss",
                    "ImprovementVsModel",
                    "ImprovementVsMarket",
                ]
            ].sort_values("ImprovementVsModel", ascending=False)
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Market": "Mercado",
            "AlphaModel": st.column_config.NumberColumn(
                "Peso do modelo",
                format="%.2%",
            ),
            "AlphaMarket": st.column_config.NumberColumn(
                "Peso do mercado",
                format="%.2%",
            ),
            "ValidationLogLoss": st.column_config.NumberColumn(
                "Erro final",
                format="%.4f",
            ),
            "ModelLogLoss": st.column_config.NumberColumn(
                "Erro do modelo",
                format="%.4f",
            ),
            "MarketLogLoss": st.column_config.NumberColumn(
                "Erro do mercado",
                format="%.4f",
            ),
            "ImprovementVsModel": st.column_config.NumberColumn(
                "Melhora vs modelo",
                format="%.4f",
            ),
            "ImprovementVsMarket": st.column_config.NumberColumn(
                "Melhora vs mercado",
                format="%.4f",
            ),
        },
    )


def render_ml_learning_page() -> None:
    """Renderiza diagnosticos de aprendizado do modelo."""
    st.title("Apostasbot | Como o modelo aprende")
    st.caption(
        "Uma leitura mais visual e simples do que pesa no modelo, se as "
        "chances fazem sentido e quando o mercado ajuda."
    )
    render_pipeline_runner()

    (
        feature_importance,
        tuning,
        probability_blend,
        calibration_curve,
        calibration_metrics,
    ) = load_ml_learning_artifacts()
    if (
        feature_importance.empty
        and tuning.empty
        and probability_blend.empty
        and calibration_curve.empty
        and calibration_metrics.empty
    ):
        st.info("Rode o pipeline com tuning XGBoost para gerar os artefatos.")
        return

    feature_importance = enrich_feature_importance(feature_importance)
    calibration_metrics = enrich_calibration_metrics(calibration_metrics)
    render_learning_overview(
        feature_importance,
        tuning,
        probability_blend,
        calibration_metrics,
    )
    st.divider()

    tabs = st.tabs(
        [
            "O que o modelo observa",
            "Chance x realidade",
            "Modelo x mercado",
            "Ajustes do motor",
            "Tabelas completas",
        ]
    )
    with tabs[0]:
        render_feature_learning(feature_importance)
    with tabs[1]:
        render_calibration_learning(calibration_curve, calibration_metrics)
    with tabs[2]:
        render_probability_blend_learning(probability_blend)
    with tabs[3]:
        render_tuning_learning(tuning)
    with tabs[4]:
        raw_tabs = st.tabs(
            [
                "Sinais",
                "Ajustes",
                "Modelo x mercado",
                "Curvas",
                "Metricas",
            ]
        )
        with raw_tabs[0]:
            st.subheader("Importancia completa")
            st.dataframe(
                feature_importance,
                width="stretch",
                hide_index=True,
            )
        with raw_tabs[1]:
            st.subheader("Tuning completo")
            st.dataframe(
                tuning,
                width="stretch",
                hide_index=True,
            )
        with raw_tabs[2]:
            st.subheader("Blend modelo x mercado")
            st.dataframe(
                enrich_probability_blend(probability_blend),
                width="stretch",
                hide_index=True,
            )
        with raw_tabs[3]:
            st.subheader("Curvas de calibracao")
            st.dataframe(
                enrich_calibration_curve(calibration_curve),
                width="stretch",
                hide_index=True,
            )
        with raw_tabs[4]:
            st.subheader("Metricas de calibracao")
            st.dataframe(
                calibration_metrics,
                width="stretch",
                hide_index=True,
            )


def max_odd_label(value: float) -> str:
    """Formata odd maxima para tabelas de regras."""
    if pd.isna(value):
        return "Sem limite"
    return f"{value:.2f}"


def filter_rule_status(row: pd.Series) -> str:
    """Classifica a regra otimizada pela avaliacao posterior."""
    if row["EvalBets"] < 10:
        return "Pouco volume"
    if row["EvalROI"] > 0:
        return "Liberada"
    if row["TuneROI"] > 0 and row["EvalROI"] <= 0:
        return "Overfit"
    return "Bloqueada"


def enrich_filter_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Adiciona campos de leitura visual ao resumo de filtros."""
    enriched = data.copy()
    enriched["Status"] = enriched.apply(filter_rule_status, axis=1)
    enriched["RoiGap"] = enriched["EvalROI"] - enriched["TuneROI"]
    enriched["MaxOddLabel"] = enriched["MaxOdd"].map(max_odd_label)
    enriched["RuleText"] = (
        "Edge "
        + enriched["EdgeThreshold"].map(lambda value: f"{value:.1%}")
        + " | Prob "
        + enriched["MinModelProb"].map(lambda value: f"{value:.1%}")
        + " | Odd "
        + enriched["MaxOddLabel"]
    )
    return enriched


def enrich_filter_grid(data: pd.DataFrame) -> pd.DataFrame:
    """Adiciona campos auxiliares a grade de filtros."""
    if data.empty:
        return data

    enriched = data.copy()
    enriched["Status"] = enriched.apply(filter_rule_status, axis=1)
    enriched["RoiGap"] = enriched["EvalROI"] - enriched["TuneROI"]
    enriched["MaxOddLabel"] = enriched["MaxOdd"].map(max_odd_label)
    return enriched


def render_filter_status_table(data: pd.DataFrame) -> None:
    """Mostra regras finais com status de uso futuro."""
    table = data[
        [
            "Status",
            "Market",
            "RuleText",
            "TuneBets",
            "TuneROI",
            "EvalBets",
            "EvalROI",
            "RoiGap",
            "EvalHitRate",
            "EvalTotalProfit",
        ]
    ].sort_values(["Status", "EvalROI"], ascending=[True, False])

    st.subheader("Regras finais")
    st.dataframe(
        style_dashboard_table(table),
        width="stretch",
        hide_index=True,
        column_config={
            "Status": "Status",
            "Market": "Mercado",
            "RuleText": "Filtro",
            "TuneBets": st.column_config.NumberColumn("Apostas ajuste", format="%.0f"),
            "TuneROI": st.column_config.NumberColumn("ROI ajuste", format="%.3f"),
            "EvalBets": st.column_config.NumberColumn("Apostas aval.", format="%.0f"),
            "EvalROI": st.column_config.NumberColumn("ROI aval.", format="%.3f"),
            "RoiGap": st.column_config.NumberColumn("Gap ROI", format="%.3f"),
            "EvalHitRate": st.column_config.NumberColumn(
                "Acerto aval.",
                format="%.3f",
            ),
            "EvalTotalProfit": st.column_config.NumberColumn(
                "Lucro aval.",
                format="R$ %.2f",
            ),
        },
    )


def render_filter_grid_view(grid: pd.DataFrame, markets: list[str]) -> None:
    """Mostra uma exploracao visual da grade de filtros."""
    if grid.empty or not markets:
        return

    selected_market = st.selectbox(
        "Mercado da grade",
        options=markets,
    )
    market_grid = grid[grid["Market"].eq(selected_market)].copy()
    if market_grid.empty:
        return

    qualified_grid = market_grid[market_grid["Qualified"]].copy()
    plot_data = qualified_grid if not qualified_grid.empty else market_grid

    left, right = st.columns((1.1, 1.0))
    with left:
        fig = px.scatter(
            plot_data,
            x="MinModelProb",
            y="EdgeThreshold",
            size="EvalBets",
            color="EvalROI",
            hover_data=[
                "MaxOddLabel",
                "TuneROI",
                "TuneBets",
                "EvalROI",
                "EvalBets",
                "EvalTotalProfit",
            ],
            labels={
                "MinModelProb": "Probabilidade minima",
                "EdgeThreshold": "Edge minimo",
                "EvalROI": "ROI avaliacao",
                "EvalBets": "Apostas aval.",
            },
            title="Mapa de filtros",
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        top_eval = plot_data.sort_values(
            ["EvalROI", "EvalTotalProfit", "EvalBets"],
            ascending=[False, False, False],
            kind="mergesort",
        ).head(15)
        st.subheader("Top por avaliacao")
        st.dataframe(
            style_dashboard_table(
                top_eval[
                    [
                        "Status",
                        "EdgeThreshold",
                        "MinModelProb",
                        "MaxOddLabel",
                        "TuneBets",
                        "TuneROI",
                        "EvalBets",
                        "EvalROI",
                        "EvalTotalProfit",
                    ]
                ]
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "Status": "Status",
                "EdgeThreshold": st.column_config.NumberColumn(
                    "Edge",
                    format="%.3f",
                ),
                "MinModelProb": st.column_config.NumberColumn(
                    "Prob",
                    format="%.3f",
                ),
                "MaxOddLabel": "Odd max",
                "TuneBets": st.column_config.NumberColumn("Ajuste", format="%.0f"),
                "TuneROI": st.column_config.NumberColumn(
                    "ROI ajuste",
                    format="%.3f",
                ),
                "EvalBets": st.column_config.NumberColumn("Aval.", format="%.0f"),
                "EvalROI": st.column_config.NumberColumn("ROI aval.", format="%.3f"),
                "EvalTotalProfit": st.column_config.NumberColumn(
                    "Lucro aval.",
                    format="R$ %.2f",
                ),
            },
        )


def render_filter_optimization_page() -> None:
    """Renderiza a busca de filtros por mercado."""
    st.title("Apostasbot | Otimizacao de filtros")
    st.caption("Busca de edge, probabilidade minima e odd maxima por mercado.")
    render_pipeline_runner()

    summary, grid = load_filter_optimization()
    if summary.empty:
        st.info("Rode o pipeline com otimizacao de filtros ativada.")
        return

    st.sidebar.header("Filtros")
    markets = sorted(summary["Market"].dropna().unique().tolist())
    selected_markets = st.sidebar.multiselect(
        "Mercados",
        options=markets,
        default=markets,
    )
    filtered_summary = enrich_filter_summary(
        summary[summary["Market"].isin(selected_markets)].copy()
    )
    filtered_grid = (
        enrich_filter_grid(grid[grid["Market"].isin(selected_markets)].copy())
        if not grid.empty
        else pd.DataFrame()
    )

    if filtered_summary.empty:
        st.info("Nenhum mercado selecionado.")
        return

    released_count = int(filtered_summary["Status"].eq("Liberada").sum())
    overfit_count = int(filtered_summary["Status"].eq("Overfit").sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Liberadas",
        f"{released_count:,}".replace(",", "."),
    )
    col2.metric(
        "Overfit",
        f"{overfit_count:,}".replace(",", "."),
    )
    col3.metric(
        "ROI aval. medio",
        format_pct(float(filtered_summary["EvalROI"].mean())),
    )
    col4.metric(
        "Lucro aval.",
        format_money(float(filtered_summary["EvalTotalProfit"].sum())),
    )

    render_filter_status_table(filtered_summary)

    left, right = st.columns(2)
    with left:
        long_roi = filtered_summary.melt(
            id_vars=["Market"],
            value_vars=["TuneROI", "EvalROI"],
            var_name="Janela",
            value_name="ROI",
        )
        long_roi["Janela"] = long_roi["Janela"].map(
            {"TuneROI": "Ajuste", "EvalROI": "Avaliacao"}
        )
        fig = px.bar(
            long_roi,
            x="Market",
            y="ROI",
            color="Janela",
            barmode="group",
            labels={"Market": "Mercado", "ROI": "ROI"},
            title="ROI ajuste vs avaliacao",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.scatter(
            filtered_summary,
            x="TuneROI",
            y="EvalROI",
            size="EvalBets",
            color="Status",
            hover_data=[
                "Market",
                "RuleText",
                "EdgeThreshold",
                "MinModelProb",
                "TuneBets",
                "EvalBets",
                "EvalTotalProfit",
            ],
            labels={
                "TuneROI": "ROI ajuste",
                "EvalROI": "ROI avaliacao",
                "EvalBets": "Apostas avaliacao",
                "Status": "Status",
            },
            title="Estabilidade da regra",
        )
        x_min = float(filtered_summary["TuneROI"].min()) - 0.02
        x_max = float(filtered_summary["TuneROI"].max()) + 0.02
        fig.add_shape(
            type="line",
            x0=x_min,
            y0=x_min,
            x1=x_max,
            y1=x_max,
            line={"dash": "dot", "color": "gray"},
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, width="stretch")

    table = filtered_summary[
        [
            "Status",
            "Market",
            "RuleText",
            "EdgeThreshold",
            "MinModelProb",
            "MaxOdd",
            "TuneBets",
            "TuneROI",
            "TuneHitRate",
            "TuneTotalProfit",
            "EvalBets",
            "EvalROI",
            "EvalHitRate",
            "EvalTotalProfit",
            "EvalAvgOdd",
        ]
    ].sort_values(["Status", "EvalROI"], ascending=[True, False])
    st.subheader("Detalhe das regras")
    st.dataframe(
        style_dashboard_table(table),
        width="stretch",
        hide_index=True,
        column_config={
            "Status": "Status",
            "Market": "Mercado",
            "RuleText": "Filtro",
            "EdgeThreshold": st.column_config.NumberColumn(
                "Edge min",
                format="%.3f",
            ),
            "MinModelProb": st.column_config.NumberColumn(
                "Prob min",
                format="%.3f",
            ),
            "MaxOdd": st.column_config.NumberColumn("Odd max", format="%.2f"),
            "TuneBets": st.column_config.NumberColumn("Apostas ajuste", format="%.0f"),
            "TuneROI": st.column_config.NumberColumn("ROI ajuste", format="%.3f"),
            "TuneHitRate": st.column_config.NumberColumn(
                "Acerto ajuste",
                format="%.3f",
            ),
            "TuneTotalProfit": st.column_config.NumberColumn(
                "Lucro ajuste",
                format="R$ %.2f",
            ),
            "EvalBets": st.column_config.NumberColumn("Apostas aval.", format="%.0f"),
            "EvalROI": st.column_config.NumberColumn("ROI aval.", format="%.3f"),
            "EvalHitRate": st.column_config.NumberColumn(
                "Acerto aval.",
                format="%.3f",
            ),
            "EvalTotalProfit": st.column_config.NumberColumn(
                "Lucro aval.",
                format="R$ %.2f",
            ),
            "EvalAvgOdd": st.column_config.NumberColumn("Odd aval.", format="%.2f"),
        },
    )

    if filtered_grid.empty:
        return

    render_filter_grid_view(filtered_grid, selected_markets)


def render_realistic_backtest_page() -> None:
    """Renderiza backtest com validacao por temporada e gestao de banca."""
    st.title("Apostasbot | Backtest realista")
    st.caption(
        "Treino, validacao e teste por temporada, grid de thresholds, "
        "stake fixa e Kelly fracionado."
    )
    render_pipeline_runner()

    summary, grid, bets, monthly, league = load_realistic_backtest()
    if summary.empty:
        st.info("Rode o pipeline com Backtest realista ativado para gerar os CSVs.")
        return

    st.sidebar.header("Filtros")
    markets = sorted(summary["Market"].dropna().unique().tolist())
    selected_markets = st.sidebar.multiselect(
        "Mercados",
        options=markets,
        default=markets,
    )
    stake_modes = sorted(summary["StakeModeName"].dropna().unique().tolist())
    selected_stake_modes = st.sidebar.multiselect(
        "Stake",
        options=stake_modes,
        default=stake_modes,
    )

    filtered = summary[
        summary["Market"].isin(selected_markets)
        & summary["StakeModeName"].isin(selected_stake_modes)
    ].copy()
    if filtered.empty:
        st.info("Nenhuma estrategia encontrada para os filtros atuais.")
        return

    strategy_ids = filtered["StrategyId"].dropna().unique().tolist()
    filtered_bets = (
        bets[bets["StrategyId"].isin(strategy_ids)].copy()
        if not bets.empty and "StrategyId" in bets.columns
        else pd.DataFrame()
    )
    filtered_monthly = (
        monthly[monthly["StrategyId"].isin(strategy_ids)].copy()
        if not monthly.empty and "StrategyId" in monthly.columns
        else pd.DataFrame()
    )
    filtered_league = (
        league[league["StrategyId"].isin(strategy_ids)].copy()
        if not league.empty and "StrategyId" in league.columns
        else pd.DataFrame()
    )

    total_staked = float(filtered["TestTotalStaked"].sum())
    total_profit = float(filtered["TestTotalProfit"].sum())
    roi = total_profit / total_staked if total_staked else 0.0
    hit_rate = (
        float(
            (filtered["TestHitRate"] * filtered["TestBets"]).sum()
            / filtered["TestBets"].sum()
        )
        if filtered["TestBets"].sum() > 0
        else 0.0
    )
    cols = st.columns(5)
    cols[0].metric("Apostas teste", f"{filtered['TestBets'].sum():.0f}")
    cols[1].metric("Lucro teste", format_money(total_profit))
    cols[2].metric("ROI teste", format_pct(roi))
    cols[3].metric("Acerto teste", format_pct(hit_rate))
    cols[4].metric(
        "Max drawdown",
        format_money(float(filtered["TestMaxDrawdown"].max())),
        delta=f"reds {int(filtered['TestLongestRedStreak'].max())}",
    )

    st.subheader("Estrategias escolhidas pela validacao")
    st.dataframe(
        style_dashboard_table(
            filtered[
                [
                    "Market",
                    "StakeModeName",
                    "EdgeThreshold",
                    "MinModelProb",
                    "MaxOddLabel",
                    "ValBets",
                    "ValROI",
                    "ValTotalProfit",
                    "TestBets",
                    "TestROI",
                    "TestTotalProfit",
                    "TestMaxDrawdown",
                    "TestLongestRedStreak",
                    "TestHitRate",
                    "TestAvgOdd",
                ]
            ]
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Market": "Mercado",
            "StakeModeName": "Stake",
            "EdgeThreshold": st.column_config.NumberColumn("Edge", format="%.3f"),
            "MinModelProb": st.column_config.NumberColumn("Prob min", format="%.3f"),
            "MaxOddLabel": "Odd max",
            "ValBets": st.column_config.NumberColumn("Apostas val.", format="%.0f"),
            "ValROI": st.column_config.NumberColumn("ROI val.", format="%.3f"),
            "ValTotalProfit": st.column_config.NumberColumn(
                "Lucro val.",
                format="R$ %.2f",
            ),
            "TestBets": st.column_config.NumberColumn("Apostas teste", format="%.0f"),
            "TestROI": st.column_config.NumberColumn("ROI teste", format="%.3f"),
            "TestTotalProfit": st.column_config.NumberColumn(
                "Lucro teste",
                format="R$ %.2f",
            ),
            "TestMaxDrawdown": st.column_config.NumberColumn(
                "Drawdown",
                format="R$ %.2f",
            ),
            "TestLongestRedStreak": st.column_config.NumberColumn(
                "Seq. reds",
                format="%.0f",
            ),
            "TestHitRate": st.column_config.NumberColumn("Acerto", format="%.3f"),
            "TestAvgOdd": st.column_config.NumberColumn("Odd media", format="%.2f"),
        },
    )

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            filtered,
            x="Market",
            y="TestROI",
            color="StakeModeName",
            barmode="group",
            hover_data=["TestBets", "TestTotalProfit", "TestMaxDrawdown"],
            labels={
                "Market": "Mercado",
                "TestROI": "ROI teste",
                "StakeModeName": "Stake",
            },
            title="ROI final por mercado",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    with right:
        fig = px.scatter(
            filtered,
            x="ValROI",
            y="TestROI",
            size="TestBets",
            color="StakeModeName",
            hover_data=["Market", "EdgeThreshold", "MinModelProb", "MaxOddLabel"],
            labels={
                "ValROI": "ROI validacao",
                "TestROI": "ROI teste",
                "StakeModeName": "Stake",
            },
            title="Validacao vs teste",
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.add_vline(x=0, line_dash="dot", line_color="gray")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    if not filtered_bets.empty:
        st.subheader("Curva de lucro e drawdown")
        curve_col, dd_col = st.columns((1.4, 1.0))
        with curve_col:
            fig = px.line(
                filtered_bets.sort_values("MatchDatetime"),
                x="MatchDatetime",
                y="CumulativeProfit",
                color="StrategyId",
                labels={
                    "MatchDatetime": "Data",
                    "CumulativeProfit": "Lucro acumulado",
                    "StrategyId": "Estrategia",
                },
                title="Lucro acumulado no teste",
            )
            st.plotly_chart(fig, width="stretch")
        with dd_col:
            fig = px.line(
                filtered_bets.sort_values("MatchDatetime"),
                x="MatchDatetime",
                y="Drawdown",
                color="StrategyId",
                labels={
                    "MatchDatetime": "Data",
                    "Drawdown": "Drawdown",
                    "StrategyId": "Estrategia",
                },
                title="Drawdown ao longo do teste",
            )
            st.plotly_chart(fig, width="stretch")

    if not filtered_monthly.empty:
        st.subheader("Lucro por mes")
        fig = px.bar(
            filtered_monthly.sort_values("Month"),
            x="Month",
            y="TotalProfit",
            color="StrategyId",
            hover_data=["Bets", "ROI", "HitRate"],
            labels={"Month": "Mes", "TotalProfit": "Lucro"},
            title="Resultado mensal no teste",
        )
        st.plotly_chart(fig, width="stretch")

    if not filtered_league.empty:
        st.subheader("Lucro por liga")
        league_plot = filtered_league.copy()
        if "LigaNome" not in league_plot.columns and "Liga" in league_plot.columns:
            league_plot["LigaNome"] = league_plot["Liga"].map(league_label)
        fig = px.bar(
            league_plot.sort_values("TotalProfit", ascending=False),
            x="LigaNome",
            y="TotalProfit",
            color="ROI",
            hover_data=["Market", "StakeModeName", "Bets", "HitRate", "AvgOdd"],
            labels={"LigaNome": "Liga", "TotalProfit": "Lucro"},
            title="Resultado por campeonato",
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            style_dashboard_table(
                league_plot[
                    [
                        "Market",
                        "StakeModeName",
                        "LigaNome",
                        "Bets",
                        "TotalProfit",
                        "ROI",
                        "HitRate",
                        "AvgOdd",
                        "MaxDrawdown",
                        "LongestRedStreak",
                    ]
                ].sort_values("TotalProfit", ascending=False)
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "Market": "Mercado",
                "StakeModeName": "Stake",
                "LigaNome": "Liga",
                "Bets": st.column_config.NumberColumn("Apostas", format="%.0f"),
                "TotalProfit": st.column_config.NumberColumn(
                    "Lucro",
                    format="R$ %.2f",
                ),
                "ROI": st.column_config.NumberColumn("ROI", format="%.3f"),
                "HitRate": st.column_config.NumberColumn("Acerto", format="%.3f"),
                "AvgOdd": st.column_config.NumberColumn("Odd media", format="%.2f"),
                "MaxDrawdown": st.column_config.NumberColumn(
                    "Drawdown",
                    format="R$ %.2f",
                ),
                "LongestRedStreak": st.column_config.NumberColumn(
                    "Seq. reds",
                    format="%.0f",
                ),
            },
        )

    if not grid.empty:
        st.subheader("Grid de thresholds")
        grid_filtered = grid[
            grid["Market"].isin(selected_markets)
            & grid["StakeModeName"].isin(selected_stake_modes)
        ].copy()
        grid_filtered = grid_filtered.sort_values(
            ["ValROI", "ValTotalProfit", "ValBets"],
            ascending=[False, False, False],
            kind="mergesort",
        ).head(250)
        st.dataframe(
            style_dashboard_table(
                grid_filtered[
                    [
                        "Market",
                        "StakeModeName",
                        "EdgeThreshold",
                        "MinModelProb",
                        "MaxOdd",
                        "Qualified",
                        "ValBets",
                        "ValROI",
                        "ValTotalProfit",
                        "TestBets",
                        "TestROI",
                        "TestTotalProfit",
                        "TestMaxDrawdown",
                        "TestLongestRedStreak",
                    ]
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def render_upcoming_page() -> None:
    """Renderiza a pagina de palpites futuros."""
    st.title("Apostasbot | Palpites futuros +EV")
    st.caption("Jogos futuros, probabilidades do modelo e odds da casa usada.")
    render_upcoming_runner()

    predictions, odds, context = load_upcoming_data()
    if not context.empty:
        summary = context.iloc[0]
        requested = str(summary.get("RequestedBookmaker", "Melhor disponivel"))
        message = str(summary.get("Message", "")).strip()
        if message:
            if bool(summary.get("UsesPreferredBookmaker", False)):
                st.info(f"Casa usada nos palpites: {requested}. {message}")
            else:
                st.info(message)
    if predictions.empty:
        if context.empty:
            st.info("Gere os palpites futuros pelo painel lateral para criar os CSVs.")
        else:
            st.warning("Nenhum palpite disponivel com a casa escolhida no momento.")
        return

    st.sidebar.header("Regra +EV")
    use_generated_rule = st.sidebar.checkbox(
        "Usar regra gerada no CSV",
        value=True,
    )
    if use_generated_rule:
        recalculated = use_generated_upcoming_value(predictions)
    else:
        edge = st.sidebar.slider(
            "Edge minimo futuro",
            min_value=0.0,
            max_value=0.20,
            value=0.05,
            step=0.005,
            format="%.3f",
        )
        min_probability = st.sidebar.slider(
            "Probabilidade minima futura",
            min_value=0.0,
            max_value=0.90,
            value=0.50,
            step=0.01,
            format="%.2f",
        )
        max_odd = st.sidebar.number_input(
            "Odd maxima futura (0 desativa)",
            min_value=0.0,
            max_value=20.0,
            value=2.50,
            step=0.05,
        )
        recalculated = recalculate_upcoming_value(
            predictions,
            edge=edge,
            min_probability=min_probability,
            max_odd=max_odd,
        )
    filtered = apply_upcoming_filters(recalculated)

    render_upcoming_metrics(filtered)
    st.divider()
    render_upcoming_charts(filtered)
    st.divider()
    render_upcoming_table(filtered)
    st.divider()
    render_bookmaker_odds(filtered, odds)


def render_backtests_page() -> None:
    """Renderiza a pagina historica de backtests."""
    st.title("Apostasbot | Backtests +EV")
    st.caption("Analise visual dos mercados Over/Under 2.5 e Resultado Final.")
    render_pipeline_runner()

    market = st.sidebar.radio(
        "Mercado",
        options=["Over 2.5", "Under 2.5", "Resultado 1X2", "Vitoria Casa/Fora"],
        horizontal=False,
    )

    data = load_market_data(market)
    filtered = apply_sidebar_filters(
        data,
        key_prefix=f"backtest_{market.replace(' ', '_').replace('/', '_')}",
    )

    st.sidebar.header("Regra de aposta")
    default_edge = 0.05
    default_prob = {
        "Over 2.5": 0.55,
        "Under 2.5": 0.55,
        "Resultado 1X2": 0.48,
        "Vitoria Casa/Fora": 0.50,
    }[market]
    default_max_odd = 1.80 if market in {"Over 2.5", "Under 2.5"} else 2.50

    edge = st.sidebar.slider(
        "Edge minimo",
        min_value=0.0,
        max_value=0.20,
        value=default_edge,
        step=0.005,
        format="%.3f",
    )
    min_probability = st.sidebar.slider(
        "Probabilidade minima",
        min_value=0.0,
        max_value=0.90,
        value=default_prob,
        step=0.01,
        format="%.2f",
    )
    max_odd = st.sidebar.number_input(
        "Odd maxima (0 desativa)",
        min_value=0.0,
        max_value=20.0,
        value=default_max_odd,
        step=0.05,
    )
    stake = st.sidebar.number_input(
        "Stake fixa",
        min_value=1.0,
        max_value=1000.0,
        value=10.0,
        step=1.0,
    )

    simulated = simulate_bets(
        filtered,
        edge=edge,
        min_probability=min_probability,
        max_odd=max_odd,
        stake=stake,
    )
    summary = build_summary(simulated)

    render_metric_row(summary)
    st.divider()
    render_charts(simulated)
    st.divider()
    render_league_summary_table(simulated)
    st.divider()
    render_table(simulated)


def main() -> None:
    """Renderiza o dashboard."""
    page = st.sidebar.radio(
        "Tela",
        options=[
            "Testes historicos",
            "Palpites futuros",
            "Comparar modelos",
            "Melhores filtros",
            "Onde funciona melhor",
            "Teste realista",
            "Aprendizado do modelo",
        ],
        horizontal=False,
    )
    if page == "Palpites futuros":
        render_upcoming_page()
    elif page == "Comparar modelos":
        render_model_comparison_page()
    elif page == "Melhores filtros":
        render_filter_optimization_page()
    elif page == "Onde funciona melhor":
        render_model_strength_page()
    elif page == "Teste realista":
        render_realistic_backtest_page()
    elif page == "Aprendizado do modelo":
        render_ml_learning_page()
    else:
        render_backtests_page()


if __name__ == "__main__":
    main()
