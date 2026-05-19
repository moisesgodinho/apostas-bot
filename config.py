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
NO_VIG_UNDER_COL = "NoVig_Prob_Under25"
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
    markets: Sequence[str] = field(
        default_factory=lambda: ["over25", "under25", "result"]
    )
    raw_dir: Path = Path("raw_data")
    output_dir: Path = Path("outputs")
    lineup_features_path: Path | None = Path("raw_data/lineup_features.csv")
    understat_xg_dir: Path = Path("raw_data/understat_xg")
    use_understat_xg: bool = True
    force_refresh_understat_xg: bool = False
    rolling_window: int = 5
    train_size: float = 0.80
    split_strategy: str = "season"
    validation_seasons: int = 1
    test_seasons: int = 1
    calibration_size: float = 0.20
    calibration_method: str = "sigmoid"
    walk_forward_splits: int = 5
    stake: float = 10.0
    kelly_bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    max_kelly_fraction: float = 0.03
    edge: float = 0.05
    min_model_prob: float = 0.55
    max_over_odd: float | None = 1.80
    min_under_prob: float = 0.55
    max_under_odd: float | None = 1.80
    min_result_prob: float = 0.48
    max_result_odd: float | None = 2.50
    min_win_prob: float = 0.50
    max_win_odd: float | None = 2.50
    elo_initial: float = 1500.0
    elo_k_factor: float = 20.0
    elo_home_advantage: float = 65.0
    run_model_comparison: bool = True
    run_filter_optimization: bool = True
    run_realistic_backtest: bool = True
    filter_optimization_train_size: float = 0.60
    min_optimization_bets: int = 30
    use_optimized_filters_for_upcoming: bool = True
    min_optimized_eval_roi: float = 0.0
    min_optimized_eval_bets: int = 10
    feature_profile: str = "extended"
    xgb_tuning_trials: int = 0
    xgb_tuning_validation_size: float = 0.20
