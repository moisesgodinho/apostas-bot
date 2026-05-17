"""Configuracao e constantes do projeto Apostasbot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DEFAULT_LEAGUES = [
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
    "BRA",
]
EXTRA_LEAGUE_URLS = {
    "BRA": "https://www.football-data.co.uk/new/BRA.csv",
}
DEFAULT_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526"]
TARGET_COL = "Over25"
RESULT_TARGET_COL = "ResultTarget"
RESULT_LABELS = ["H", "D", "A"]
RESULT_NAME_MAP = {
    "H": "Casa",
    "D": "Empate",
    "A": "Fora",
}
ODD_OVER_COL = "Odd_Over25"
ODD_UNDER_COL = "Odd_Under25"
ODD_HOME_COL = "Odd_Home"
ODD_DRAW_COL = "Odd_Draw"
ODD_AWAY_COL = "Odd_Away"
RAW_IMPLIED_OVER_COL = "Raw_Implied_Prob_Over25"
RAW_IMPLIED_UNDER_COL = "Raw_Implied_Prob_Under25"
NO_VIG_OVER_COL = "NoVig_Prob_Over25"
OVERROUND_COL = "Overround_OverUnder25"
RAW_IMPLIED_HOME_COL = "Raw_Implied_Prob_Home"
RAW_IMPLIED_DRAW_COL = "Raw_Implied_Prob_Draw"
RAW_IMPLIED_AWAY_COL = "Raw_Implied_Prob_Away"
NO_VIG_HOME_COL = "NoVig_Prob_Home"
NO_VIG_DRAW_COL = "NoVig_Prob_Draw"
NO_VIG_AWAY_COL = "NoVig_Prob_Away"
OVERROUND_1X2_COL = "Overround_1X2"


@dataclass(frozen=True)
class PipelineConfig:
    """Configuracao principal da esteira de modelagem."""

    leagues: Sequence[str]
    seasons: Sequence[str]
    markets: Sequence[str] = field(default_factory=lambda: ["over25", "result"])
    raw_dir: Path = Path("raw_data")
    output_dir: Path = Path("outputs")
    rolling_window: int = 5
    train_size: float = 0.80
    calibration_size: float = 0.20
    calibration_method: str = "sigmoid"
    walk_forward_splits: int = 5
    stake: float = 10.0
    edge: float = 0.05
    min_model_prob: float = 0.55
    max_over_odd: float | None = 1.80
    min_result_prob: float = 0.48
    max_result_odd: float | None = 2.50
    min_win_prob: float = 0.50
    max_win_odd: float | None = 2.50
