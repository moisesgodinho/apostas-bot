"""Interface de linha de comando do pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_LEAGUES, DEFAULT_SEASONS, PipelineConfig


def parse_args() -> PipelineConfig:
    """Le argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description=(
            "Baixa dados do football-data.co.uk, treina um modelo para "
            "Over/Under 2.5 gols e executa um backtest de apostas +EV."
        )
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=DEFAULT_LEAGUES,
        help="Codigos das ligas. Exemplo: --leagues E0 SP1",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=DEFAULT_SEASONS,
        help="Temporadas no padrao football-data. Exemplo: 2324 2425 2526",
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=["over25", "result", "win", "all"],
        default=["all"],
        help="Mercados para executar: over25, result, win ou all.",
    )
    parser.add_argument(
        "--raw-dir",
        default="raw_data",
        help="Pasta local para armazenar os CSVs baixados.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Pasta para salvar resultados do backtest.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Numero de jogos anteriores usados nas medias moveis.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.80,
        help="Percentual inicial da serie temporal usado para treino.",
    )
    parser.add_argument(
        "--calibration-size",
        type=float,
        default=0.20,
        help=(
            "Percentual final do treino usado para calibrar probabilidades "
            "sem olhar o teste."
        ),
    )
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Metodo de calibracao de probabilidades.",
    )
    parser.add_argument(
        "--walk-forward-splits",
        type=int,
        default=5,
        help="Numero de janelas walk-forward. Use 0 para desativar.",
    )
    parser.add_argument(
        "--stake",
        type=float,
        default=10.0,
        help="Valor fixo simulado por aposta.",
    )
    parser.add_argument(
        "--edge",
        type=float,
        default=0.05,
        help="Margem minima de valor: prob_modelo - prob_implicita.",
    )
    parser.add_argument(
        "--min-model-prob",
        type=float,
        default=0.55,
        help=(
            "Probabilidade minima do modelo para simular aposta. Aumente "
            "para melhorar taxa de acerto e reduzir volume."
        ),
    )
    parser.add_argument(
        "--max-over-odd",
        type=float,
        default=1.80,
        help=(
            "Odd maxima permitida no Over 2.5. Use para evitar apostas "
            "muito improvaveis; use 0 para desativar."
        ),
    )
    parser.add_argument(
        "--min-result-prob",
        type=float,
        default=0.48,
        help="Probabilidade minima do modelo para apostar no mercado 1X2.",
    )
    parser.add_argument(
        "--max-result-odd",
        type=float,
        default=2.50,
        help="Odd maxima permitida no 1X2. Use 0 para desativar.",
    )
    parser.add_argument(
        "--min-win-prob",
        type=float,
        default=0.50,
        help="Probabilidade minima do modelo para apostar em vitoria casa/fora.",
    )
    parser.add_argument(
        "--max-win-odd",
        type=float,
        default=2.50,
        help="Odd maxima permitida no mercado de vitoria. Use 0 para desativar.",
    )

    args = parser.parse_args()
    max_over_odd = args.max_over_odd if args.max_over_odd > 0 else None
    max_result_odd = args.max_result_odd if args.max_result_odd > 0 else None
    max_win_odd = args.max_win_odd if args.max_win_odd > 0 else None
    markets = ["over25", "result", "win"] if "all" in args.markets else args.markets
    return PipelineConfig(
        leagues=args.leagues,
        seasons=args.seasons,
        markets=markets,
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        rolling_window=args.rolling_window,
        train_size=args.train_size,
        calibration_size=args.calibration_size,
        calibration_method=args.calibration_method,
        walk_forward_splits=args.walk_forward_splits,
        stake=args.stake,
        edge=args.edge,
        min_model_prob=args.min_model_prob,
        max_over_odd=max_over_odd,
        min_result_prob=args.min_result_prob,
        max_result_odd=max_result_odd,
        min_win_prob=args.min_win_prob,
        max_win_odd=max_win_odd,
    )
