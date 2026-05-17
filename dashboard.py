"""Dashboard local para analisar backtests de apostas +EV.

Execute com:
    streamlit run dashboard.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import DEFAULT_LEAGUES, DEFAULT_SEASONS


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OVER25_PATH = OUTPUT_DIR / "backtest_over25_results.csv"
RESULT_1X2_PATH = OUTPUT_DIR / "backtest_result_1x2_results.csv"
WIN_PATH = OUTPUT_DIR / "backtest_win_results.csv"
UPCOMING_PATH = OUTPUT_DIR / "upcoming_predictions.csv"
UPCOMING_ODDS_PATH = OUTPUT_DIR / "upcoming_odds_by_bookmaker.csv"
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


def league_label(code: str) -> str:
    """Retorna o nome amigavel da liga."""
    return LEAGUE_NAME_MAP.get(code, code)


st.set_page_config(
    page_title="Apostasbot | Backtests +EV",
    page_icon="A",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(path: Path, modified_at: float) -> pd.DataFrame:
    """Carrega um CSV de backtest com datas normalizadas."""
    _ = modified_at
    data = pd.read_csv(path)
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
        data = pd.read_csv(path)
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
    return f"{value:.2%}"


def render_pipeline_runner() -> None:
    """Renderiza controles para atualizar backtests pelo dashboard."""
    with st.sidebar.expander("Atualizar backtests", expanded=False):
        st.caption("Roda o pipeline local e atualiza os CSVs em outputs.")
        market = st.selectbox(
            "Mercados",
            options=["all", "over25", "result", "win"],
            format_func={
                "all": "Todos",
                "over25": "Over 2.5",
                "result": "Resultado 1X2",
                "win": "Vitoria Casa/Fora",
            }.get,
        )
        leagues = st.multiselect(
            "Ligas do treino",
            options=DEFAULT_LEAGUES,
            default=DEFAULT_LEAGUES,
            format_func=league_label,
        )
        seasons = st.multiselect(
            "Temporadas",
            options=DEFAULT_SEASONS,
            default=DEFAULT_SEASONS,
        )
        walk_forward_splits = st.number_input(
            "Folds walk-forward",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Use 0 para uma atualizacao mais rapida.",
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
            "--walk-forward-splits",
            str(walk_forward_splits),
        ]

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
            "Usa fixtures publicas do Football-Data. Brasil ainda precisa "
            "de outra fonte de jogos futuros."
        )
        market = st.selectbox(
            "Mercados dos palpites",
            options=["all", "over25", "result", "win"],
            format_func={
                "all": "Todos",
                "over25": "Over 2.5",
                "result": "Resultado 1X2",
                "win": "Vitoria Casa/Fora",
            }.get,
        )
        fixture_leagues = [league for league in DEFAULT_LEAGUES if league != "BRA"]
        leagues = st.multiselect(
            "Ligas",
            options=DEFAULT_LEAGUES,
            default=fixture_leagues,
            format_func=league_label,
        )
        seasons = st.multiselect(
            "Temporadas historicas",
            options=DEFAULT_SEASONS,
            default=DEFAULT_SEASONS,
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
            "--days-ahead",
            str(days_ahead),
        ]
        if force_refresh:
            command.append("--force-refresh-fixtures")

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


def apply_sidebar_filters(data: pd.DataFrame) -> pd.DataFrame:
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

    leagues = sorted(data["Liga"].dropna().unique().tolist())
    selected_leagues = st.sidebar.multiselect(
        "Ligas",
        options=leagues,
        default=leagues,
        format_func=league_label,
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


def render_table(data: pd.DataFrame) -> None:
    """Mostra tabela das apostas selecionadas."""
    bets = data[data["SimBet"]].copy()
    if bets.empty:
        return

    table = bets[
        [
            "MatchDatetime",
            "LigaNome",
            "HomeTeam",
            "AwayTeam",
            "Selection",
            "SelectionOdd",
            "ModelProb",
            "NoVigProb",
            "MarketEdge",
            "Hit",
            "SimProfit",
        ]
    ].sort_values("MatchDatetime", ascending=False)

    st.subheader("Apostas simuladas")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "MatchDatetime": st.column_config.DatetimeColumn("Data"),
            "LigaNome": "Liga",
            "HomeTeam": "Mandante",
            "AwayTeam": "Visitante",
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
            "MarketEdge": st.column_config.NumberColumn("Edge", format="%.2f"),
            "Hit": "Acertou",
            "SimProfit": st.column_config.NumberColumn("Lucro", format="R$ %.2f"),
        },
    )


def load_upcoming_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega palpites futuros e odds por casa."""
    if not UPCOMING_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    predictions = load_optional_csv(UPCOMING_PATH, UPCOMING_PATH.stat().st_mtime)
    if UPCOMING_ODDS_PATH.exists():
        odds = load_optional_csv(
            UPCOMING_ODDS_PATH,
            UPCOMING_ODDS_PATH.stat().st_mtime,
        )
    else:
        odds = pd.DataFrame()
    return predictions, odds


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

    leagues = sorted(data["Liga"].dropna().unique().tolist())
    selected_leagues = st.sidebar.multiselect(
        "Ligas dos palpites",
        options=leagues,
        default=leagues,
        format_func=league_label,
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


def render_upcoming_table(data: pd.DataFrame) -> None:
    """Mostra tabela principal de palpites futuros."""
    if data.empty:
        st.info("Nenhum palpite encontrado com os filtros atuais.")
        return

    table = data.sort_values(
        ["UiValueBet", "MatchDatetime", "Edge"],
        ascending=[False, True, False],
    )[
        [
            "MatchDatetimeBR",
            "LigaNome",
            "HomeTeam",
            "AwayTeam",
            "Market",
            "Selection",
            "BestBookmaker",
            "BestOdd",
            "ModelProb",
            "ImpliedProb",
            "Edge",
            "UiValueBet",
        ]
    ]

    st.subheader("Palpites futuros")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "MatchDatetimeBR": st.column_config.DatetimeColumn("Data (BR)"),
            "LigaNome": "Liga",
            "HomeTeam": "Mandante",
            "AwayTeam": "Visitante",
            "Market": "Mercado",
            "Selection": "Selecao",
            "BestBookmaker": "Melhor casa",
            "BestOdd": st.column_config.NumberColumn("Melhor odd", format="%.2f"),
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
            "Edge": st.column_config.NumberColumn("Edge", format="%.3f"),
            "UiValueBet": "+EV",
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
                "BestBookmaker",
                "BestOdd",
            ],
            labels={
                "ModelProb": "Probabilidade do modelo",
                "Edge": "Edge",
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
    st.dataframe(
        pivot,
        width="stretch",
        hide_index=True,
    )


def render_upcoming_page() -> None:
    """Renderiza a pagina de palpites futuros."""
    st.title("Apostasbot | Palpites futuros +EV")
    st.caption("Jogos futuros, probabilidades do modelo e odds por casa.")
    render_upcoming_runner()

    predictions, odds = load_upcoming_data()
    if predictions.empty:
        st.info("Gere os palpites futuros pelo painel lateral para criar os CSVs.")
        return

    st.sidebar.header("Regra +EV")
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
    st.caption("Analise visual dos mercados Over 2.5 e Resultado Final 1X2.")
    render_pipeline_runner()

    market = st.sidebar.radio(
        "Mercado",
        options=["Over 2.5", "Resultado 1X2", "Vitoria Casa/Fora"],
        horizontal=False,
    )

    data = load_market_data(market)
    filtered = apply_sidebar_filters(data)

    st.sidebar.header("Regra de aposta")
    default_edge = 0.05
    default_prob = {
        "Over 2.5": 0.55,
        "Resultado 1X2": 0.48,
        "Vitoria Casa/Fora": 0.50,
    }[market]
    default_max_odd = 1.80 if market == "Over 2.5" else 2.50

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
    render_table(simulated)


def main() -> None:
    """Renderiza o dashboard."""
    page = st.sidebar.radio(
        "Pagina",
        options=["Backtests", "Palpites futuros"],
        horizontal=False,
    )
    if page == "Palpites futuros":
        render_upcoming_page()
    else:
        render_backtests_page()


if __name__ == "__main__":
    main()
