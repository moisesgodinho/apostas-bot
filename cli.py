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
        choices=["over25", "under25", "result", "win", "all"],
        default=["all"],
        help="Mercados para executar: over25, under25, result, win ou all.",
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
        "--lineup-features-path",
        default="raw_data/lineup_features.csv",
        help=(
            "CSV opcional com escalações/desfalques por time. Use vazio "
            "para desativar."
        ),
    )
    parser.add_argument(
        "--understat-xg-dir",
        default="raw_data/understat_xg",
        help="Pasta de cache para xG historico do Understat.",
    )
    parser.add_argument(
        "--no-understat-xg",
        action="store_true",
        help="Desativa a coleta/uso de xG do Understat.",
    )
    parser.add_argument(
        "--force-refresh-understat-xg",
        action="store_true",
        help="Forca novo download do cache de xG do Understat.",
    )
    parser.add_argument(
        "--use-clubelo",
        action="store_true",
        help=(
            "Usa ClubElo como fonte externa de forca dos times. Quando "
            "indisponivel, o Elo interno continua sendo usado."
        ),
    )
    parser.add_argument(
        "--clubelo-cache-dir",
        default="raw_data/clubelo",
        help="Pasta de cache para historicos ClubElo por clube.",
    )
    parser.add_argument(
        "--force-refresh-clubelo",
        action="store_true",
        help="Forca novo download do cache ClubElo.",
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
        "--split-strategy",
        choices=["season", "chronological"],
        default="season",
        help=(
            "Como separar treino/validacao/teste. 'season' usa temporadas "
            "inteiras por liga; 'chronological' usa corte percentual simples."
        ),
    )
    parser.add_argument(
        "--validation-seasons",
        type=int,
        default=1,
        help="Quantidade de temporadas por liga usadas para validacao.",
    )
    parser.add_argument(
        "--test-seasons",
        type=int,
        default=1,
        help="Quantidade de temporadas por liga usadas para teste final.",
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
        "--walk-forward-initial-train-fraction",
        type=float,
        default=0.50,
        help=(
            "Percentual inicial de datas usado para o primeiro treino "
            "walk-forward."
        ),
    )
    parser.add_argument(
        "--walk-forward-min-test-rows",
        type=int,
        default=200,
        help="Minimo de jogos exigido em cada bloco de teste walk-forward.",
    )
    parser.add_argument(
        "--no-time-decay-weights",
        action="store_true",
        help="Desativa pesos temporais no treino do XGBoost.",
    )
    parser.add_argument(
        "--time-decay-half-life-days",
        type=float,
        default=540.0,
        help=(
            "Meia-vida dos pesos temporais em dias. Valores menores favorecem "
            "mais os jogos recentes."
        ),
    )
    parser.add_argument(
        "--min-time-decay-weight",
        type=float,
        default=0.20,
        help="Peso minimo bruto antes da normalizacao temporal.",
    )
    parser.add_argument(
        "--stake",
        type=float,
        default=10.0,
        help="Valor fixo simulado por aposta.",
    )
    parser.add_argument(
        "--kelly-bankroll",
        type=float,
        default=1000.0,
        help="Banca inicial simulada para Kelly fracionado.",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fator fracionario aplicado sobre a stake de Kelly cheia.",
    )
    parser.add_argument(
        "--max-kelly-fraction",
        type=float,
        default=0.03,
        help="Teto de exposicao por aposta no Kelly fracionado.",
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
        "--min-under-prob",
        type=float,
        default=0.55,
        help="Probabilidade minima do modelo para apostar no Under 2.5.",
    )
    parser.add_argument(
        "--max-under-odd",
        type=float,
        default=1.80,
        help="Odd maxima permitida no Under 2.5. Use 0 para desativar.",
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
    parser.add_argument(
        "--elo-initial",
        type=float,
        default=1500.0,
        help="Rating inicial usado para times sem historico Elo.",
    )
    parser.add_argument(
        "--elo-k-factor",
        type=float,
        default=20.0,
        help="Fator K do Elo; valores maiores reagem mais rapido a resultados.",
    )
    parser.add_argument(
        "--elo-home-advantage",
        type=float,
        default=65.0,
        help="Bonus de rating aplicado ao mandante no calculo Elo.",
    )
    parser.add_argument(
        "--skip-model-comparison",
        action="store_true",
        help="Nao treina a comparacao automatica com odds vs sem odds.",
    )
    parser.add_argument(
        "--skip-filter-optimization",
        action="store_true",
        help="Nao executa a busca automatica de filtros por mercado.",
    )
    parser.add_argument(
        "--skip-realistic-backtest",
        action="store_true",
        help="Nao gera o backtest realista por temporada/stake.",
    )
    parser.add_argument(
        "--filter-optimization-train-size",
        type=float,
        default=0.60,
        help=(
            "Percentual inicial do periodo de teste usado para escolher "
            "filtros. O restante avalia os filtros escolhidos."
        ),
    )
    parser.add_argument(
        "--min-optimization-bets",
        type=int,
        default=30,
        help="Minimo de apostas na janela de ajuste para aceitar uma regra.",
    )
    parser.add_argument(
        "--feature-profile",
        choices=["base", "extended"],
        default="extended",
        help=(
            "Perfil de features do modelo. 'base' usa sinais estaveis; "
            "'extended' inclui estatisticas de jogo, forma, descanso, "
            "importancia e movimento de odds."
        ),
    )
    parser.add_argument(
        "--xgb-tuning-trials",
        type=int,
        default=0,
        help=(
            "Numero de perfis XGBoost testados em validacao temporal. "
            "Use 0 para desativar."
        ),
    )
    parser.add_argument(
        "--xgb-tuning-validation-size",
        type=float,
        default=0.20,
        help="Percentual do treino base usado para escolher hiperparametros.",
    )

    args = parser.parse_args()
    max_over_odd = args.max_over_odd if args.max_over_odd > 0 else None
    max_under_odd = args.max_under_odd if args.max_under_odd > 0 else None
    max_result_odd = args.max_result_odd if args.max_result_odd > 0 else None
    max_win_odd = args.max_win_odd if args.max_win_odd > 0 else None
    markets = (
        ["over25", "under25", "result", "win"]
        if "all" in args.markets
        else args.markets
    )
    return PipelineConfig(
        leagues=args.leagues,
        seasons=args.seasons,
        markets=markets,
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        lineup_features_path=(
            Path(args.lineup_features_path)
            if args.lineup_features_path
            else None
        ),
        understat_xg_dir=Path(args.understat_xg_dir),
        use_understat_xg=not args.no_understat_xg,
        force_refresh_understat_xg=args.force_refresh_understat_xg,
        clubelo_cache_dir=Path(args.clubelo_cache_dir),
        use_clubelo=args.use_clubelo,
        force_refresh_clubelo=args.force_refresh_clubelo,
        rolling_window=args.rolling_window,
        train_size=args.train_size,
        split_strategy=args.split_strategy,
        validation_seasons=args.validation_seasons,
        test_seasons=args.test_seasons,
        calibration_size=args.calibration_size,
        calibration_method=args.calibration_method,
        walk_forward_splits=args.walk_forward_splits,
        walk_forward_initial_train_fraction=(
            args.walk_forward_initial_train_fraction
        ),
        walk_forward_min_test_rows=args.walk_forward_min_test_rows,
        use_time_decay_weights=not args.no_time_decay_weights,
        time_decay_half_life_days=args.time_decay_half_life_days,
        min_time_decay_weight=args.min_time_decay_weight,
        stake=args.stake,
        kelly_bankroll=args.kelly_bankroll,
        kelly_fraction=args.kelly_fraction,
        max_kelly_fraction=args.max_kelly_fraction,
        edge=args.edge,
        min_model_prob=args.min_model_prob,
        max_over_odd=max_over_odd,
        min_under_prob=args.min_under_prob,
        max_under_odd=max_under_odd,
        min_result_prob=args.min_result_prob,
        max_result_odd=max_result_odd,
        min_win_prob=args.min_win_prob,
        max_win_odd=max_win_odd,
        elo_initial=args.elo_initial,
        elo_k_factor=args.elo_k_factor,
        elo_home_advantage=args.elo_home_advantage,
        run_model_comparison=not args.skip_model_comparison,
        run_filter_optimization=not args.skip_filter_optimization,
        run_realistic_backtest=not args.skip_realistic_backtest,
        filter_optimization_train_size=args.filter_optimization_train_size,
        min_optimization_bets=args.min_optimization_bets,
        feature_profile=args.feature_profile,
        xgb_tuning_trials=args.xgb_tuning_trials,
        xgb_tuning_validation_size=args.xgb_tuning_validation_size,
    )
