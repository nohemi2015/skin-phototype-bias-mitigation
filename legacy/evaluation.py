
"""
evaluation.py
============================================================
Evaluación binaria para dos dominios:
- Brown Skin (dominio objetivo)
- HAM10000 (dominio fuente)

Calcula:
AUC-ROC, accuracy, sensitivity, specificity, precision, F1,
umbral óptimo por Youden, índice de Youden, matriz de confusión
y Domain Gap = AUC_HAM - AUC_BS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)

ArrayLike = Union[np.ndarray, list, tuple]


def predict_scores(model, dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Obtiene etiquetas reales y probabilidades predichas."""
    y_true_parts = []

    for _, labels in dataset:
        y_true_parts.append(np.asarray(labels).reshape(-1))

    if not y_true_parts:
        raise ValueError("El dataset está vacío.")

    y_true = np.concatenate(y_true_parts).astype(int)
    y_score = np.asarray(
        model.predict(dataset, verbose=0)
    ).reshape(-1).astype(float)

    if len(y_true) != len(y_score):
        raise ValueError(
            f"Etiquetas y predicciones no coinciden: "
            f"{len(y_true)} vs {len(y_score)}"
        )

    return y_true, y_score


def optimal_youden_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
) -> Tuple[float, float]:
    """Devuelve el umbral que maximiza J = sensibilidad + especificidad - 1."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if np.unique(y_true).size < 2:
        raise ValueError("Se requieren ambas clases para calcular ROC y Youden.")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    valid = np.isfinite(thresholds)
    fpr = fpr[valid]
    tpr = tpr[valid]
    thresholds = thresholds[valid]

    if thresholds.size == 0:
        raise ValueError("No se encontraron umbrales ROC válidos.")

    youden_values = tpr - fpr
    best_index = int(np.argmax(youden_values))

    return (
        float(thresholds[best_index]),
        float(youden_values[best_index]),
    )


def evaluate_predictions(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calcula métricas binarias.

    Si threshold=None, usa el umbral óptimo de Youden.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if len(y_true) == 0 or len(y_true) != len(y_score):
        raise ValueError("Datos vacíos o tamaños incompatibles.")

    if np.unique(y_true).size < 2:
        raise ValueError("Se requieren ambas clases para calcular AUC.")

    auc = float(roc_auc_score(y_true, y_score))

    if threshold is None:
        threshold, _ = optimal_youden_threshold(y_true, y_score)

    threshold = float(threshold)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = float(
        precision_score(y_true, y_pred, zero_division=0)
    )

    f1 = float(
        f1_score(y_true, y_pred, zero_division=0)
    )

    accuracy = float(
        accuracy_score(y_true, y_pred)
    )

    youden = float(
        sensitivity + specificity - 1.0
    )

    return {
        "auc": auc,
        "accuracy": accuracy,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": precision,
        "f1": f1,
        "threshold": threshold,
        "youden": youden,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_model(
    model,
    dataset,
    threshold: Optional[float] = None,
):
    """Evalúa un modelo sobre un único dataset."""
    y_true, y_score = predict_scores(model, dataset)
    metrics = evaluate_predictions(
        y_true,
        y_score,
        threshold=threshold,
    )
    return metrics, y_true, y_score


def evaluate_two_domains(
    model,
    ham_dataset,
    bs_dataset,
    threshold_source: str = "bs",
) -> Dict[str, Any]:
    """
    Evalúa el mismo modelo sobre HAM10000 y Brown Skin.

    Por compatibilidad con el pipeline:
    - Las métricas sin prefijo corresponden a Brown Skin.
    - También se guardan auc_bs, auc_ham y domain_gap.
    - Se añaden todas las métricas HAM con prefijo ham_.
    """
    threshold_source = threshold_source.lower()

    if threshold_source not in {"bs", "ham", "independent"}:
        raise ValueError(
            "threshold_source debe ser 'bs', 'ham' o 'independent'."
        )

    y_true_ham, y_score_ham = predict_scores(model, ham_dataset)
    y_true_bs, y_score_bs = predict_scores(model, bs_dataset)

    if threshold_source == "bs":
        shared_threshold, _ = optimal_youden_threshold(
            y_true_bs,
            y_score_bs,
        )
        bs_metrics = evaluate_predictions(
            y_true_bs,
            y_score_bs,
            threshold=shared_threshold,
        )
        ham_metrics = evaluate_predictions(
            y_true_ham,
            y_score_ham,
            threshold=shared_threshold,
        )

    elif threshold_source == "ham":
        shared_threshold, _ = optimal_youden_threshold(
            y_true_ham,
            y_score_ham,
        )
        bs_metrics = evaluate_predictions(
            y_true_bs,
            y_score_bs,
            threshold=shared_threshold,
        )
        ham_metrics = evaluate_predictions(
            y_true_ham,
            y_score_ham,
            threshold=shared_threshold,
        )

    else:
        bs_metrics = evaluate_predictions(
            y_true_bs,
            y_score_bs,
        )
        ham_metrics = evaluate_predictions(
            y_true_ham,
            y_score_ham,
        )

    # Métricas principales = Brown Skin
    result = dict(bs_metrics)

    result.update({
        "auc_bs": float(bs_metrics["auc"]),
        "auc_ham": float(ham_metrics["auc"]),
        "domain_gap": float(
            ham_metrics["auc"] - bs_metrics["auc"]
        ),
    })

    for key, value in ham_metrics.items():
        result[f"ham_{key}"] = value

    return result


def save_predictions(
    output_path: Union[str, Path],
    y_true: ArrayLike,
    y_score: ArrayLike,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Guarda etiquetas, probabilidades y metadatos en CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(y_true) != len(y_score):
        raise ValueError(
            "y_true y y_score deben tener la misma longitud."
        )

    df = pd.DataFrame({
        "y_true": y_true.astype(int),
        "y_score": y_score.astype(float),
    })

    if metadata:
        for key, value in metadata.items():
            df[key] = value

    df.to_csv(output_path, index=False)
    return output_path


def add_youden_column(
    df: pd.DataFrame,
    sensitivity_col: str = "sensitivity",
    specificity_col: str = "specificity",
    output_col: str = "youden",
) -> pd.DataFrame:
    """Agrega Youden a una tabla existente."""
    result = df.copy()
    result[output_col] = (
        result[sensitivity_col]
        + result[specificity_col]
        - 1.0
    )
    return result


def add_domain_gap_column(
    df: pd.DataFrame,
    auc_ham_col: str = "auc_ham",
    auc_bs_col: str = "auc_bs",
    output_col: str = "domain_gap",
) -> pd.DataFrame:
    """Agrega Domain Gap = AUC_HAM - AUC_BS."""
    result = df.copy()
    result[output_col] = (
        result[auc_ham_col]
        - result[auc_bs_col]
    )
    return result


def summarize_results(
    results_source: Union[str, Path, pd.DataFrame],
    group_col: str = "model",
) -> pd.DataFrame:
    """Resume métricas por modelo: media, std y count."""
    if isinstance(results_source, pd.DataFrame):
        df = results_source.copy()
    else:
        df = pd.read_csv(results_source)

    if df.empty:
        raise ValueError("El archivo de resultados está vacío.")

    metric_candidates = [
        "auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "threshold",
        "youden",
        "auc_bs",
        "auc_ham",
        "domain_gap",
        "ham_accuracy",
        "ham_sensitivity",
        "ham_specificity",
        "ham_precision",
        "ham_f1",
        "ham_threshold",
        "ham_youden",
    ]

    metrics = [
        col for col in metric_candidates
        if col in df.columns
    ]

    if not metrics:
        raise ValueError(
            "No se encontraron columnas métricas reconocidas."
        )

    return (
        df.groupby(group_col)[metrics]
        .agg(["mean", "std", "count"])
        .sort_index()
    )
