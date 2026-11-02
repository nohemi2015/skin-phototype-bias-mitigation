"""
generar_roc_youden.py

Recupera las predicciones por muestra de los modelos YA ENTRENADOS
(guardados en model_file dentro de resultados_modelA_ddi.csv) volviendo
a generar exactamente los mismos folds de validacion (misma semilla,
mismos parametros de RepeatedStratifiedKFold que entrenar_modelA_ddi.py)
y recargando cada modelo para predecir sobre su fold correspondiente.

No vuelve a entrenar nada -- solo recarga y predice, así que es rapido.

Genera, por backbone:
    - predicciones_<backbone>.csv        (y_true, y_score, repeat, fold -- todas las corridas apiladas)
    - roc_<backbone>.png                 (curva ROC real, pooled sobre las 15 corridas)
    - resumen_youden_<backbone>.csv      (umbral y J de Youden por fold + el agregado)

IMPORTANTE sobre la curva "pooled": cada repeticion (de las 3) usa una
particion distinta, así que un mismo paciente/imagen puede aparecer en
distintos folds de validacion a traves de las repeticiones (aunque
nunca dos veces en el MISMO repeat). Esto es valido para visualizar la
curva agregada, pero si vas a reportar un intervalo de confianza sobre
ella, ten presente que las observaciones no son completamente
independientes entre repeticiones.

Uso:
    python generar_roc_youden.py --results_csv resultados/modelo_A_ddi/resultados_modelA_ddi.csv --out_dir ./roc_output
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

from config import DDI_DIR
from dataset import collect_image_paths, make_dataset
from evaluation import optimal_youden_threshold, save_predictions

N_SPLITS = 5
N_REPEATS = 3
BASE_SEED = 2026


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
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Reconstruyendo el dataset DDI (mismo orden que en el entrenamiento)...")
    ddi_df = collect_image_paths(DDI_DIR, "DDI")
    val_splits = make_repeated_splits(ddi_df)

    results = pd.read_csv(args.results_csv)

    for backbone, group in results.groupby("backbone"):
        print(f"\n=== {backbone} ===")
        all_true, all_score, all_repeat, all_fold = [], [], [], []
        youden_rows = []

        for _, row in group.iterrows():
            repeat, fold = int(row["repeat"]), int(row["fold"])
            model_path = row["model_file"]

            if not os.path.exists(model_path):
                print(f"  [aviso] no encontrado, se omite: {model_path}")
                continue

            val_idx = val_splits[(repeat, fold)]
            val_df = ddi_df.iloc[val_idx].reset_index(drop=True)
            val_ds = make_dataset(val_df, augment=False, shuffle=False)

            model = keras.models.load_model(model_path)
            y_score = np.asarray(model.predict(val_ds, verbose=0)).reshape(-1)
            y_true = val_df["label"].values.astype(int)

            all_true.append(y_true)
            all_score.append(y_score)
            all_repeat.append(np.full(len(y_true), repeat))
            all_fold.append(np.full(len(y_true), fold))

            thr, j = optimal_youden_threshold(y_true, y_score)
            youden_rows.append({
                "backbone": backbone, "repeat": repeat, "fold": fold,
                "threshold_youden": thr, "youden_index": j,
                "auc_fold": roc_auc_score(y_true, y_score),
            })

            keras.backend.clear_session()
            print(f"  ok: rep{repeat} fold{fold} (n={len(y_true)})")

        if not all_true:
            print(f"  [aviso] no se recuperaron predicciones para {backbone}, se omite")
            continue

        y_true_all = np.concatenate(all_true)
        y_score_all = np.concatenate(all_score)
        repeat_all = np.concatenate(all_repeat)
        fold_all = np.concatenate(all_fold)

        # Guardar predicciones por muestra (esto es lo que faltaba para
        # tener curvas ROC reales, no ajustadas)
        preds_path = os.path.join(args.out_dir, f"predicciones_{backbone}.csv")
        save_predictions(
            preds_path, y_true_all, y_score_all,
            metadata={"backbone": backbone, "repeat": repeat_all, "fold": fold_all},
        )
        print(f"  Predicciones guardadas: {preds_path}")

        # Youden agregado (sobre todas las predicciones pooled)
        thr_pooled, j_pooled = optimal_youden_threshold(y_true_all, y_score_all)
        auc_pooled = roc_auc_score(y_true_all, y_score_all)

        youden_df = pd.DataFrame(youden_rows)
        youden_summary_path = os.path.join(args.out_dir, f"resumen_youden_{backbone}.csv")
        youden_df.to_csv(youden_summary_path, index=False)

        print(f"  Youden por fold -> media: {youden_df['youden_index'].mean():.4f} "
              f"(std: {youden_df['youden_index'].std():.4f})")
        print(f"  Youden pooled (todas las predicciones juntas): {j_pooled:.4f} "
              f"| umbral: {thr_pooled:.4f} | AUC pooled: {auc_pooled:.4f}")

        # Curva ROC real (pooled)
        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"{backbone} (AUC={auc_pooled:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC=0.500)")

        # Marca el punto de operacion en el umbral de Youden pooled
        y_pred_at_thr = (y_score_all >= thr_pooled).astype(int)
        tp = np.sum((y_pred_at_thr == 1) & (y_true_all == 1))
        fn = np.sum((y_pred_at_thr == 0) & (y_true_all == 1))
        fp_ = np.sum((y_pred_at_thr == 1) & (y_true_all == 0))
        tn = np.sum((y_pred_at_thr == 0) & (y_true_all == 0))
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp_) if (tn + fp_) > 0 else np.nan
        ax.scatter([1 - spec], [sens], color="red", zorder=5,
                   label=f"Umbral Youden (sens={sens:.2f}, spec={spec:.2f})")

        ax.set_xlabel("1 - Especificidad (FPR)")
        ax.set_ylabel("Sensibilidad (TPR)")
        ax.set_title(f"Curva ROC real (pooled) -- {backbone} sobre DDI")
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        roc_path = os.path.join(args.out_dir, f"roc_{backbone}.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"  Curva ROC guardada: {roc_path}")

    print(f"\nListo. Todo en: {args.out_dir}")


if __name__ == "__main__":
    main()
