"""Treinamento, calibracao e metricas de modelos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, precision_score

from config import RESULT_LABELS


def train_xgboost_model(x_train: pd.DataFrame, y_train: pd.Series):
    """Instancia e treina um XGBClassifier."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost nao esta instalado. Rode: "
            "python -m pip install -r requirements.txt"
        ) from exc

    num_classes = int(y_train.nunique())
    model_params = {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
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

    model = XGBClassifier(**model_params)
    model.fit(x_train, y_train)
    return model


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
    base_model = train_xgboost_model(x_model_train, y_model_train)
    return calibrate_prefit_model(
        base_model,
        x_calibration,
        y_calibration,
        calibration_method,
    )


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
