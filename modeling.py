"""Treinamento, calibracao e metricas de modelos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
)

from config import RESULT_LABELS


DEFAULT_XGB_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}
XGB_TUNING_GRID = [
    {},
    {
        "n_estimators": 300,
        "learning_rate": 0.04,
        "max_depth": 2,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 2.0,
        "reg_lambda": 1.5,
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.025,
        "max_depth": 3,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 3.0,
        "reg_lambda": 2.0,
    },
    {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.75,
        "colsample_bytree": 0.85,
        "min_child_weight": 2.0,
        "reg_lambda": 2.5,
        "reg_alpha": 0.05,
    },
    {
        "n_estimators": 650,
        "learning_rate": 0.02,
        "max_depth": 2,
        "subsample": 0.95,
        "colsample_bytree": 0.75,
        "min_child_weight": 4.0,
        "reg_lambda": 3.0,
        "reg_alpha": 0.05,
    },
    {
        "n_estimators": 350,
        "learning_rate": 0.035,
        "max_depth": 3,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
        "min_child_weight": 5.0,
        "reg_lambda": 4.0,
        "reg_alpha": 0.10,
    },
]
# Mantem sinal minimo do modelo para preservar capacidade de encontrar +EV.
BLEND_ALPHA_GRID = np.linspace(0.25, 1.0, 16)


def build_xgboost_params(
    y_train: pd.Series,
    overrides: dict[str, float | int] | None = None,
) -> dict[str, float | int | str]:
    """Monta parametros do XGBoost conforme o tipo do alvo."""
    num_classes = int(y_train.nunique())
    model_params = DEFAULT_XGB_PARAMS.copy()
    if overrides:
        model_params.update(overrides)

    if num_classes > 2:
        model_params.update(
            {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": num_classes,
            }
        )
    else:
        model_params.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            }
        )
    return model_params


def train_xgboost_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, float | int] | None = None,
):
    """Instancia e treina um XGBClassifier."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost nao esta instalado. Rode: "
            "python -m pip install -r requirements.txt"
        ) from exc

    model = XGBClassifier(**build_xgboost_params(y_train, params))
    model.fit(x_train, y_train)
    return model


def temporal_train_validation_split(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino e validacao para tuning sem embaralhar a serie."""
    if not 0.0 < validation_size < 0.5:
        raise ValueError("validation_size deve estar entre 0 e 0.5.")

    split_index = int(len(x_train) * (1.0 - validation_size))
    if split_index == 0 or split_index == len(x_train):
        raise ValueError("Dados insuficientes para criar janela de validacao.")

    x_fit = x_train.iloc[:split_index].copy()
    x_validation = x_train.iloc[split_index:].copy()
    y_fit = y_train.iloc[:split_index].copy()
    y_validation = y_train.iloc[split_index:].copy()
    return x_fit, x_validation, y_fit, y_validation


def tune_xgboost_hyperparameters(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    max_trials: int,
    validation_size: float,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Escolhe hiperparametros por LogLoss em uma janela temporal."""
    if max_trials <= 0:
        return {}, pd.DataFrame()

    try:
        x_fit, x_validation, y_fit, y_validation = temporal_train_validation_split(
            x_train,
            y_train,
            validation_size,
        )
    except ValueError as exc:
        print(f"[tuning] Ignorado: {exc}")
        return {}, pd.DataFrame()

    expected_classes = sorted(y_train.unique().tolist())
    if sorted(y_fit.unique().tolist()) != expected_classes:
        print("[tuning] Ignorado: treino interno nao contem todas as classes.")
        return {}, pd.DataFrame()

    trials = XGB_TUNING_GRID[: max(1, min(max_trials, len(XGB_TUNING_GRID)))]
    trial_rows: list[dict[str, float | int | str]] = []
    best_score = np.inf
    best_params: dict[str, float | int] = {}

    print(
        "[tuning] XGBoost: "
        f"{len(trials)} tentativas, validacao temporal {len(x_validation):,} jogos"
    )
    for trial_number, params in enumerate(trials, start=1):
        model = train_xgboost_model(x_fit, y_fit, params)
        probabilities = model.predict_proba(x_validation)
        score = log_loss(
            y_validation,
            probabilities,
            labels=expected_classes,
        )
        row = {
            "Trial": trial_number,
            "ValidationLogLoss": score,
            **build_xgboost_params(y_train, params),
        }
        trial_rows.append(row)

        if score < best_score:
            best_score = score
            best_params = params.copy()

    tuning_summary = pd.DataFrame(trial_rows).sort_values(
        "ValidationLogLoss",
        kind="mergesort",
    )
    print(f"[tuning] Melhor LogLoss validacao: {best_score:.4f}")
    return best_params, tuning_summary


def temporal_train_calibration_split(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    calibration_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino e calibracao preservando a ordem cronologica."""
    if not 0.0 < calibration_size < 0.5:
        raise ValueError("calibration_size deve estar entre 0 e 0.5.")

    split_index = int(len(x_train) * (1.0 - calibration_size))
    if split_index == 0 or split_index == len(x_train):
        raise ValueError("Dados insuficientes para criar janela de calibracao.")

    x_model_train = x_train.iloc[:split_index].copy()
    x_calibration = x_train.iloc[split_index:].copy()
    y_model_train = y_train.iloc[:split_index].copy()
    y_calibration = y_train.iloc[split_index:].copy()

    if y_model_train.nunique() < 2 or y_calibration.nunique() < 2:
        raise ValueError(
            "A janela de treino/calibracao precisa conter as duas classes."
        )

    return x_model_train, x_calibration, y_model_train, y_calibration


def calibrate_prefit_model(
    model,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    method: str,
):
    """Calibra um modelo ja treinado, com compatibilidade entre sklearns."""
    try:
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method=method,
        )
    except ImportError:
        calibrator = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit",
        )

    calibrator.fit(x_calibration, y_calibration)
    return calibrator


def train_calibrated_xgboost_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    calibration_size: float,
    calibration_method: str,
    xgb_tuning_trials: int = 0,
    xgb_tuning_validation_size: float = 0.20,
):
    """Treina XGBoost e calibra probabilidades em janela temporal posterior."""
    (
        x_model_train,
        x_calibration,
        y_model_train,
        y_calibration,
    ) = temporal_train_calibration_split(x_train, y_train, calibration_size)

    print(
        "[modelo] Treino base/calibracao: "
        f"{len(x_model_train):,}/{len(x_calibration):,} jogos "
        f"({calibration_method})"
    )
    best_params, tuning_summary = tune_xgboost_hyperparameters(
        x_model_train,
        y_model_train,
        max_trials=xgb_tuning_trials,
        validation_size=xgb_tuning_validation_size,
    )
    base_model = train_xgboost_model(x_model_train, y_model_train, best_params)
    calibrated_model = calibrate_prefit_model(
        base_model,
        x_calibration,
        y_calibration,
        calibration_method,
    )
    calibrated_model.selected_xgb_params_ = build_xgboost_params(
        y_model_train,
        best_params,
    )
    calibrated_model.xgb_tuning_summary_ = tuning_summary
    calibrated_model.feature_names_ = list(x_train.columns)
    calibrated_model.base_feature_importances_ = getattr(
        base_model,
        "feature_importances_",
        None,
    )
    return calibrated_model


def binary_brier_score(
    y_true: pd.Series,
    probabilities: np.ndarray,
    positive_label: int = 1,
) -> float:
    """Calcula Brier Score para um evento binario especifico."""
    y_binary = pd.Series(y_true).eq(positive_label).astype(int).to_numpy()
    clipped_probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    return float(brier_score_loss(y_binary, clipped_probabilities))


def multiclass_brier_score(
    y_true: pd.Series,
    probabilities: np.ndarray,
    labels: list[int],
) -> float:
    """Calcula Brier Score multiclasses usando codificacao one-hot."""
    probability_matrix = np.asarray(probabilities, dtype=float)
    label_to_index = {label: index for index, label in enumerate(labels)}
    one_hot = np.zeros_like(probability_matrix, dtype=float)

    for row_index, label in enumerate(pd.Series(y_true).tolist()):
        class_index = label_to_index.get(label)
        if class_index is not None and class_index < one_hot.shape[1]:
            one_hot[row_index, class_index] = 1.0

    return float(np.mean(np.sum((probability_matrix - one_hot) ** 2, axis=1)))


def build_calibration_curve_frame(
    y_true: pd.Series,
    probabilities: np.ndarray,
    market: str,
    outcome: str,
    positive_label: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Agrupa previsoes por faixas de probabilidade para diagnosticar calibracao."""
    if n_bins <= 0:
        raise ValueError("n_bins deve ser maior que zero.")

    probability_values = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    observed_values = pd.Series(y_true).eq(positive_label).astype(int).to_numpy()
    if len(probability_values) != len(observed_values):
        raise ValueError("probabilities e y_true precisam ter o mesmo tamanho.")

    rows: list[dict[str, float | int | str]] = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for bin_number in range(1, n_bins + 1):
        lower = float(bin_edges[bin_number - 1])
        upper = float(bin_edges[bin_number])
        if bin_number == n_bins:
            mask = (probability_values >= lower) & (probability_values <= upper)
        else:
            mask = (probability_values >= lower) & (probability_values < upper)

        count = int(mask.sum())
        if count == 0:
            continue

        bin_probabilities = probability_values[mask]
        bin_observed = observed_values[mask]
        mean_predicted = float(bin_probabilities.mean())
        observed_rate = float(bin_observed.mean())
        calibration_gap = observed_rate - mean_predicted
        rows.append(
            {
                "Market": market,
                "Outcome": outcome,
                "Bin": bin_number,
                "BinStart": lower,
                "BinEnd": upper,
                "MeanPredictedProb": mean_predicted,
                "ObservedRate": observed_rate,
                "Count": count,
                "CalibrationGap": calibration_gap,
                "AbsCalibrationGap": abs(calibration_gap),
                "BrierScore": float(
                    np.mean((bin_probabilities - bin_observed) ** 2)
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_calibration_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """Resume Brier, ECE e vies medio a partir da curva de calibracao."""
    if curve.empty:
        return pd.DataFrame()

    group_cols = ["Market", "Outcome"]
    if "CalibrationMethod" in curve.columns:
        group_cols.append("CalibrationMethod")

    rows: list[dict[str, float | int | str]] = []
    for group_values, group in curve.groupby(group_cols, dropna=False):
        group_key = (
            group_values
            if isinstance(group_values, tuple)
            else (group_values,)
        )
        count = float(group["Count"].sum())
        if count <= 0:
            continue

        row = dict(zip(group_cols, group_key))
        row.update(
            {
                "Rows": int(count),
                "MeanPredictedProb": float(
                    np.average(group["MeanPredictedProb"], weights=group["Count"])
                ),
                "ObservedRate": float(
                    np.average(group["ObservedRate"], weights=group["Count"])
                ),
                "BrierScore": float(
                    np.average(group["BrierScore"], weights=group["Count"])
                ),
                "ECE": float(
                    np.average(group["AbsCalibrationGap"], weights=group["Count"])
                ),
                "MaxAbsGap": float(group["AbsCalibrationGap"].max()),
            }
        )
        row["CalibrationBias"] = row["ObservedRate"] - row["MeanPredictedProb"]
        rows.append(row)

    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: pd.Series,
    probabilities: np.ndarray,
    positive_label: int = 1,
    n_bins: int = 10,
) -> float:
    """Calcula Expected Calibration Error em faixas de probabilidade."""
    curve = build_calibration_curve_frame(
        y_true,
        probabilities,
        market="Modelo",
        outcome="Evento",
        positive_label=positive_label,
        n_bins=n_bins,
    )
    if curve.empty:
        return float("nan")
    return float(
        np.average(curve["AbsCalibrationGap"], weights=curve["Count"])
    )


def multiclass_expected_calibration_error(
    y_true: pd.Series,
    probabilities: np.ndarray,
    labels: list[int],
    n_bins: int = 10,
) -> float:
    """Calcula ECE medio one-vs-rest para um modelo multiclasses."""
    probability_matrix = np.asarray(probabilities, dtype=float)
    scores = []
    for class_index, label in enumerate(labels):
        if class_index >= probability_matrix.shape[1]:
            continue
        scores.append(
            expected_calibration_error(
                y_true,
                probability_matrix[:, class_index],
                positive_label=label,
                n_bins=n_bins,
            )
        )

    finite_scores = [score for score in scores if np.isfinite(score)]
    if not finite_scores:
        return float("nan")
    return float(np.mean(finite_scores))


def normalize_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    """Normaliza uma matriz de probabilidades por linha."""
    matrix = np.asarray(probabilities, dtype=float)
    matrix = np.clip(matrix, 1e-6, 1.0)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return matrix / row_sums


def blend_probabilities(
    model_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    alpha_model: float,
) -> np.ndarray:
    """Mistura probabilidades do modelo com probabilidades no-vig do mercado."""
    alpha = float(np.clip(alpha_model, 0.0, 1.0))
    model_values = np.asarray(model_probabilities, dtype=float)
    market_values = np.asarray(market_probabilities, dtype=float)
    blended = (alpha * model_values) + ((1.0 - alpha) * market_values)

    if blended.ndim == 1:
        return np.clip(blended, 1e-6, 1.0 - 1e-6)
    return normalize_probability_matrix(blended)


def _probability_logloss(
    y_true: pd.Series,
    probabilities: np.ndarray,
    labels: list[int],
) -> float:
    """Calcula LogLoss aceitando probabilidades binarias ou multiclasses."""
    probability_values = np.asarray(probabilities, dtype=float)
    if probability_values.ndim == 1:
        return float(log_loss(y_true, probability_values, labels=labels))
    return float(log_loss(y_true, probability_values, labels=labels))


def tune_probability_blend(
    y_validation: pd.Series,
    model_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    labels: list[int],
    market: str,
    alpha_grid: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    """Escolhe o peso entre XGBoost e mercado por LogLoss na validacao."""
    grid = BLEND_ALPHA_GRID if alpha_grid is None else alpha_grid
    model_logloss = _probability_logloss(
        y_validation,
        model_probabilities,
        labels,
    )
    market_logloss = _probability_logloss(
        y_validation,
        market_probabilities,
        labels,
    )

    rows: list[dict[str, float | str | bool]] = []
    best_alpha = 1.0
    best_logloss = np.inf
    for alpha_model in grid:
        blended = blend_probabilities(
            model_probabilities,
            market_probabilities,
            float(alpha_model),
        )
        blend_logloss = _probability_logloss(y_validation, blended, labels)
        rows.append(
            {
                "Market": market,
                "AlphaModel": float(alpha_model),
                "AlphaMarket": float(1.0 - alpha_model),
                "ValidationLogLoss": blend_logloss,
                "ModelLogLoss": model_logloss,
                "MarketLogLoss": market_logloss,
                "ImprovementVsModel": model_logloss - blend_logloss,
                "ImprovementVsMarket": market_logloss - blend_logloss,
                "IsBest": False,
            }
        )
        if blend_logloss < best_logloss:
            best_logloss = blend_logloss
            best_alpha = float(alpha_model)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        best_index = summary["ValidationLogLoss"].idxmin()
        summary.loc[best_index, "IsBest"] = True
        summary = summary.sort_values(
            ["Market", "ValidationLogLoss"],
            kind="mergesort",
        ).reset_index(drop=True)
    print(
        "[blend] "
        f"{market}: alpha modelo {best_alpha:.2f}, "
        f"LogLoss validacao {best_logloss:.4f}"
    )
    return best_alpha, summary


def evaluate_model(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Calcula metricas de classificacao para o conjunto de teste."""
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision_over": precision_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "logloss": log_loss(y_true, probabilities),
        "brier_score": binary_brier_score(y_true, probabilities),
        "calibration_ece": expected_calibration_error(y_true, probabilities),
    }


def evaluate_match_result_model(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calcula metricas para o modelo multiclasses de Resultado Final."""
    predictions = np.argmax(probabilities, axis=1)
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "logloss": log_loss(y_true, probabilities, labels=[0, 1, 2]),
        "brier_score": multiclass_brier_score(
            y_true,
            probabilities,
            labels=[0, 1, 2],
        ),
        "calibration_ece": multiclass_expected_calibration_error(
            y_true,
            probabilities,
            labels=[0, 1, 2],
        ),
    }

    for index, label in enumerate(RESULT_LABELS):
        metrics[f"precision_{label.lower()}"] = precision_score(
            y_true,
            predictions,
            labels=[index],
            average="micro",
            zero_division=0,
        )

    return metrics
