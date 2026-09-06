"""
generar_roc_combinado.py

Reproduce el estilo visual de las Figuras 2/3 del manuscrito original
(varias configuraciones -- A, B, C10...C80, D, E -- superpuestas en una
sola grafica ROC), pero reconstruido de PREDICCIONES REALES en vez de
curvas ajustadas matematicamente al AUC medio.

Como el pipeline de repetibilidad no guarda predicciones por muestra,
este script recarga cada modelo YA ENTRENADO (columna model_file del
CSV de resultados) y vuelve a predecir sobre su fold de validacion
correspondiente, regenerando exactamente los mismos splits (misma
semilla) que en el entrenamiento original. No se reentrena nada.

Uso:
    python generar_roc_combinado.py ^
        --results_csv resultados\\repetibilidad\\resultados_repetibilidad.csv ^
        --backbone mobilenetv2 ^
        --dataset_dir "C:\\CASEIB2026_proyecto\\datos_piel\\pieles_oscuras" ^
        --dataset_name "Brown Skin" ^
        --out roc_combinado_mobilenetv2.png

    # Para el CSV de un dominio alternativo (ej. DDI), usa su propio
    # results_csv y su carpeta de imagenes:
    python generar_roc_combinado.py ^
        --results_csv resultados\\repetibilidad_ddi\\resultados_repetibilidad_ddi.csv ^
        --backbone efficientnetb0 ^
        --dataset_dir "C:\\CASEIB2026_proyecto\\datos_piel\\DDI_clasificado" ^
        --dataset_name "DDI" ^
        --out roc_combinado_ddi_efficientnetb0.png

Nota: si tu CSV no tiene columna 'backbone' (version anterior, un solo
backbone), usa --backbone none y el script ignora ese filtro.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow import keras

from dataset import collect_image_paths, make_dataset
from evaluation import optimal_youden_threshold

N_SPLITS = 5
N_REPEATS = 3
BASE_SEED = 2026

# Orden y colores fijos para que coincida visualmente con el estilo
# del manuscrito original
MODEL_ORDER = ["A", "B", "C10", "C20", "C40", "C60", "C80", "D", "E"]
MODEL_LABELS = {
    "A": "Test A (source-only)",
    "B": "Test B (target-only)",
    "C10": "Test C10 (10 layers)",
    "C20": "Test C20 (20 layers)",
    "C40": "Test C40 (40 layers)",
    "C60": "Test C60 (60 layers)",
    "C80": "Test C80 (80 layers)",
    "D": "Test D (fusion)",
    "E": "Test E (balanced fusion)",
}


def make_repeated_splits(df):
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=BASE_SEED,
    )
    splits = {}
    for index, (train_idx, val_idx) in enumerate(splitter.split(df, df["label"])):
        repeat = index // N_SPLITS + 1
        fold = index % N_SPLITS + 1
        splits[(repeat, fold)] = val_idx
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--backbone", default="none",
                     help="mobilenetv2, efficientnetb0, o 'none' si el CSV no distingue backbone")
    ap.add_argument("--dataset_dir", required=True,
                     help="Carpeta del dominio objetivo (BENIGNAS/MALIGNAS), la misma usada al entrenar")
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None,
                     help="Lista separada por comas de configuraciones a incluir "
                          "(default: todas las que existan en el CSV)")
    args = ap.parse_args()

    results = pd.read_csv(args.results_csv)
    if args.backbone.lower() != "none" and "backbone" in results.columns:
        results = results[results["backbone"].astype(str).str.lower() == args.backbone.lower()]

    print(f"Reconstruyendo el dataset objetivo: {args.dataset_dir}")
    target_df = collect_image_paths(args.dataset_dir, args.dataset_name)
    val_splits = make_repeated_splits(target_df)

    requested = args.models.split(",") if args.models else MODEL_ORDER
    requested = [m for m in requested if m in results["model"].astype(str).unique()]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1,
            label="Random Classifier (AUC = 0.500)")

    for model_name in requested:
        group = results[results["model"].astype(str) == model_name]
        if group.empty:
            print(f"[aviso] sin filas para {model_name}, se omite")
            continue

        all_true, all_score = [], []
        for _, row in group.iterrows():
            repeat, fold = int(row["repeat"]), int(row["fold"])
            model_path = row["model_file"]
            if not os.path.exists(model_path):
                print(f"  [aviso] no encontrado: {model_path}")
                continue

            val_idx = val_splits.get((repeat, fold))
            if val_idx is None:
                print(f"  [aviso] no hay split para repeat={repeat} fold={fold}, se omite")
                continue

            val_df = target_df.iloc[val_idx].reset_index(drop=True)
            val_ds = make_dataset(val_df, augment=False, shuffle=False)

            model = keras.models.load_model(model_path)
            y_score = np.asarray(model.predict(val_ds, verbose=0)).reshape(-1)
            y_true = val_df["label"].values.astype(int)

            all_true.append(y_true)
            all_score.append(y_score)
            keras.backend.clear_session()

        if not all_true:
            print(f"[aviso] no se recuperaron predicciones para {model_name}, se omite")
            continue

        y_true_all = np.concatenate(all_true)
        y_score_all = np.concatenate(all_score)

        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
        auc = roc_auc_score(y_true_all, y_score_all)
        thr, _ = optimal_youden_threshold(y_true_all, y_score_all)

        y_pred_at_thr = (y_score_all >= thr).astype(int)
        tp = np.sum((y_pred_at_thr == 1) & (y_true_all == 1))
        fn = np.sum((y_pred_at_thr == 0) & (y_true_all == 1))
        fp_ = np.sum((y_pred_at_thr == 1) & (y_true_all == 0))
        tn = np.sum((y_pred_at_thr == 0) & (y_true_all == 0))
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp_) if (tn + fp_) > 0 else np.nan

        label = MODEL_LABELS.get(model_name, model_name)
        line, = ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {auc:.3f})")
        ax.scatter([1 - spec], [sens], color=line.get_color(), edgecolor="black",
                   zorder=5, s=60)

        print(f"{model_name}: AUC={auc:.4f}  n={len(y_true_all)}  "
              f"(sens={sens:.3f}, spec={spec:.3f} en umbral Youden)")

    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curves (real, pooled) -- {args.dataset_name} "
                 f"[{args.backbone if args.backbone.lower()!='none' else ''}]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\nGuardado: {args.out}")


if __name__ == "__main__":
    main()
