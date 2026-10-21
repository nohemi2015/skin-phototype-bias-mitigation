"""
evaluar_zeroshot_final.py  (21-oct-2026)
========================================
Evaluacion zero-shot del modelo FINAL (sin cherry-picking) entrenado con
    python entrenar_leakage_safe_v2.py --model C40 --train_final

Usa el mismo preprocesamiento que el entrenamiento (decode_image de
entrenar_leakage_safe_v2.py) y reporta, para cada cohorte externa:
  (i)  umbral CONGELADO de la validacion de Brown Skin  -> zero-shot estricto
  (ii) umbral re-optimizado (Youden) en la propia cohorte -> se reporta aparte
AUC, sensibilidad y especificidad con IC 95 % por bootstrap (B = 1000).

Debe estar en la misma carpeta que entrenar_leakage_safe_v2.py.

Uso (una vez por cohorte):
    python evaluar_zeroshot_final.py ^
        --final_json "resultados\\leakage_safe\\FINAL_C40_mobilenetv2.json" ^
        --external_dir "datos_piel\\DDI_clasificado" ^
        --dataset_name DDI

La carpeta externa debe tener subcarpetas BENIGNAS y MALIGNAS.
Salida: resultados\\leakage_safe\\zeroshot\\<dataset>\\
    predicciones_zeroshot_<dataset>.csv   (y_true, y_score, filepath)
    resumen_zeroshot_<dataset>.json
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tensorflow import keras

from entrenar_leakage_safe_v2 import (RESULTS_DIR, collect_image_paths,
                                       make_dataset, predict_scores,
                                       youden_threshold)

N_BOOT = 1000
SEED = 2026


def sens_spec(y, s, thr):
    p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    return (tp / (tp + fn) if tp + fn else 0.0,
            tn / (tn + fp) if tn + fp else 0.0)


def bootstrap(y, s, thr_frozen, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    out = {"auc": [], "sens_frozen": [], "spec_frozen": [],
           "sens_youden": [], "spec_youden": []}
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yb, sb = y[i], s[i]
        if len(np.unique(yb)) < 2:
            continue
        out["auc"].append(roc_auc_score(yb, sb))
        a, b = sens_spec(yb, sb, thr_frozen)
        out["sens_frozen"].append(a); out["spec_frozen"].append(b)
        t = youden_threshold(yb, sb)          # se re-optimiza en cada remuestreo
        a, b = sens_spec(yb, sb, t)
        out["sens_youden"].append(a); out["spec_youden"].append(b)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_json", required=True,
                    help="JSON que escribe --train_final (modelo y umbral congelado)")
    ap.add_argument("--external_dir", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    info = json.load(open(args.final_json))
    model_file, thr_frozen = info["model_file"], float(info["frozen_threshold_from_bs_val"])
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / "zeroshot" / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modelo congelado: {model_file}\nUmbral congelado (validacion BS): {thr_frozen:.4f}")
    model = keras.models.load_model(model_file)

    df = collect_image_paths(args.external_dir, args.dataset_name)
    print(f"{args.dataset_name}: {len(df)} imagenes "
          f"({int((df.label == 0).sum())} benignas, {int((df.label == 1).sum())} malignas)")
    y, s = predict_scores(model, make_dataset(df, augment=False, shuffle=False))

    pd.DataFrame({"y_true": y, "y_score": s, "filepath": df["filepath"].values}).to_csv(
        out_dir / f"predicciones_zeroshot_{args.dataset_name}.csv", index=False)

    auc = float(roc_auc_score(y, s))
    sf, pf = sens_spec(y, s, thr_frozen)
    thr_y = youden_threshold(y, s)
    sy, py = sens_spec(y, s, thr_y)
    ci = bootstrap(y, s, thr_frozen)

    summary = {
        "dataset": args.dataset_name, "n": int(len(y)),
        "n_benign": int((y == 0).sum()), "n_malignant": int((y == 1).sum()),
        "frozen_model_file": model_file,
        "auc": auc, "auc_ci95": ci["auc"],
        "frozen_threshold": thr_frozen,
        "sensitivity_frozen": sf, "sensitivity_frozen_ci95": ci["sens_frozen"],
        "specificity_frozen": pf, "specificity_frozen_ci95": ci["spec_frozen"],
        "youden_threshold_on_external": thr_y,
        "sensitivity_youden": sy, "sensitivity_youden_ci95": ci["sens_youden"],
        "specificity_youden": py, "specificity_youden_ci95": ci["spec_youden"],
        "control_bs_auc": info.get("control_bs_auc"),
        "domain_gap_vs_control": (info["control_bs_auc"] - auc)
                                 if info.get("control_bs_auc") is not None else None,
        "n_bootstrap": N_BOOT,
    }
    path = out_dir / f"resumen_zeroshot_{args.dataset_name}.json"
    json.dump(summary, open(path, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nGuardado en: {out_dir}")


if __name__ == "__main__":
    main()
