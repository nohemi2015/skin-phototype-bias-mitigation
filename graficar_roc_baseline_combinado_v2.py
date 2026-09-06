"""
graficar_roc_baseline_combinado.py

Combina en una sola grafica las curvas ROC reales de varios datasets
baseline, para una arquitectura a la vez. Acepta DOS formatos de CSV
de entrada (los detecta automaticamente):

  1. roc_A_<arch>_<dataset>.csv  -- de entrenar_<dataset>_modelA_*.py
     columnas: threshold, fpr, tpr, specificity, youden

  2. predicciones_<backbone>.csv -- de generar_roc_youden.py / evaluar_zeroshot.py
     columnas: y_true, y_score (la curva se calcula aqui mismo)

Uso: le pasas pares "Etiqueta=ruta_al_csv" separados por coma.

    python graficar_roc_baseline_combinado.py --out roc_baseline_mobilenetv2.png --curvas ^
        "HAM10000=resultados/modelo_A_ham10000/roc_A_mobilenetv2_ham10000.csv,"^
        "Brown Skin=resultados/modelo_A_bs/roc_A_mobilenetv2_bs.csv,"^
        "Fitzpatrick17k=resultados/modelo_A_fitzpatrick/roc_A_mobilenetv2_fitzpatrick.csv,"^
        "DDI=resultados/zeroshot_DDI/predicciones_mobilenetv2.csv,"^
        "DDI-2=resultados/modelo_A_ddi2/roc_A_mobilenetv2_ddi2.csv,"^
        "PAD-UFES-20=resultados/modelo_A_padufes/roc_A_mobilenetv2_padufes.csv"

(ajusta cada ruta a donde realmente tengas cada archivo -- no asumas
que coinciden con este ejemplo, verifica con "dir" primero)
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc as sk_auc


def load_curve(path):
    """Detecta el formato y devuelve (fpr, tpr, auc, idx_youden_optimo)."""
    df = pd.read_csv(path)

    if {"fpr", "tpr"}.issubset(df.columns):
        fpr, tpr = df["fpr"].values, df["tpr"].values
        order = np.argsort(fpr)
        auc_val = sk_auc(fpr[order], tpr[order])
        idx = df["youden"].idxmax() if "youden" in df.columns else int(np.argmax(tpr - fpr))
        return fpr, tpr, auc_val, idx

    elif {"y_true", "y_score"}.issubset(df.columns):
        y_true, y_score = df["y_true"].values, df["y_score"].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
        idx = int(np.argmax(tpr - fpr))
        return fpr, tpr, auc_val, idx

    else:
        raise ValueError(
            f"Formato no reconocido en {path}. Columnas encontradas: {list(df.columns)}. "
            f"Se esperaba ('fpr','tpr',...) o ('y_true','y_score')."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curvas", required=True,
                     help="Lista 'Etiqueta=ruta.csv' separadas por coma")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Baseline ROC Curves (real)")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
            label="Random Classifier (AUC = 0.500)")

    pairs = [p for p in args.curvas.split(",") if p.strip()]
    for pair in pairs:
        if "=" not in pair:
            print(f"[aviso] entrada mal formada, se omite: {pair}")
            continue
        label, path = pair.split("=", 1)
        label, path = label.strip(), path.strip()

        try:
            fpr, tpr, auc_val, idx = load_curve(path)
        except (FileNotFoundError, ValueError) as e:
            print(f"[aviso] {label}: {e}")
            continue

        line, = ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {auc_val:.3f})")
        ax.scatter([fpr[idx]], [tpr[idx]], color=line.get_color(),
                   edgecolor="black", zorder=5, s=50)
        print(f"{label}: AUC={auc_val:.4f}  (leido de {path})")

    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(args.title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\nGuardado: {args.out}")


if __name__ == "__main__":
    main()
