"""
evaluar_zeroshot.py

Experimento 2b del informe del asesor: congela el MEJOR modelo ya
entrenado (por defecto, C40 = HAM10000 -> Brown Skin con fine-tuning de
40 capas) y lo evalua SIN REENTRENAR sobre una cohorte real externa
(DDI, Fitzpatrick17k, PAD-UFES-20, DDI2...), reportando AUC, sensibilidad
y especificidad con IC 95% por bootstrap (1000 iteraciones), mas la
brecha real de dominio frente al AUC interno.

Seleccion del "mejor modelo": entre las 15 corridas (5 folds x 3
repeticiones) del modelo solicitado (--model_config, default C40), se
usa el fold con mayor AUC de validacion interna como el modelo
congelado. Esto es una interpretacion simple de "freeze the best
model"; la alternativa es promediar las probabilidades de las 15
corridas (ensamble) -- si prefieres esa variante, se puede agregar.

Requiere en la misma carpeta: config.py, dataset.py, evaluation.py
(no necesita models.py porque solo carga modelos ya entrenados).

Uso:
    python evaluar_zeroshot.py ^
        --results_csv resultados\\repetibilidad\\resultados_repetibilidad.csv ^
        --external_dir "C:\\1. DDI dataset\\DDI_clasificado" ^
        --dataset_name DDI ^
        --out_dir resultados\\zeroshot_DDI

    # Congelando el umbral interno de BS en vez de recalcularlo en el
    # externo (mas riguroso, verdaderamente "zero-shot" tambien en el
    # umbral de decision):
    python evaluar_zeroshot.py ^
        --results_csv resultados\\repetibilidad\\resultados_repetibilidad.csv ^
        --external_dir "C:\\1. DDI dataset\\DDI_clasificado" ^
        --dataset_name DDI ^
        --out_dir resultados\\zeroshot_DDI ^
        --internal_threshold 0.47
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow import keras

from dataset import collect_image_paths, make_dataset, summarize_df
from evaluation import predict_scores, optimal_youden_threshold, save_predictions

N_BOOTSTRAP_DEFAULT = 1000


def pick_best_model(results_csv, model_config):
    df = pd.read_csv(results_csv)
    subset = df[df["model"].astype(str) == model_config]

    if subset.empty:
        raise ValueError(
            f"No hay filas con model == '{model_config}' en {results_csv}"
        )

    best_row = subset.loc[subset["auc"].idxmax()]
    internal_mean_auc = subset["auc"].mean()

    print(f"Modelo '{model_config}': {len(subset)} corridas encontradas.")
    print(f"  AUC interno medio (las {len(subset)} corridas): {internal_mean_auc:.4f}")
    print(f"  Mejor corrida individual: repeat={best_row['repeat']} "
          f"fold={best_row['fold']}  AUC={best_row['auc']:.4f}")
    print(f"  Modelo congelado: {best_row['model_file']}")

    return best_row["model_file"], float(internal_mean_auc), float(best_row["auc"])


def sens_spec_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = (2 * precision * sens / (precision + sens)) if (precision and sens and (precision + sens) > 0) else np.nan
    return sens, spec, precision, f1


def bootstrap_metrics(y_true, y_prob, threshold, n_bootstrap, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, senss, specs, precs, f1s = [], [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
        s, sp, pr, f1 = sens_spec_at_threshold(yt, yp, threshold)
        senss.append(s)
        specs.append(sp)
        precs.append(pr)
        f1s.append(f1)

    def ci(arr):
        arr = np.array(arr, dtype=float)
        return {
            "mean": float(np.nanmean(arr)),
            "ci_low": float(np.nanpercentile(arr, 2.5)),
            "ci_high": float(np.nanpercentile(arr, 97.5)),
        }

    return {
        "auc": ci(aucs), "sensitivity": ci(senss), "specificity": ci(specs),
        "precision": ci(precs), "f1": ci(f1s), "n_valid_bootstrap": len(aucs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", required=True,
                     help="CSV de resultados de la repetibilidad (para localizar el mejor modelo)")
    ap.add_argument("--model_config", default="C40",
                     help="Que configuracion congelar (default: C40)")
    ap.add_argument("--external_dir", required=True,
                     help="Carpeta de la cohorte externa, con subcarpetas BENIGNAS/ y MALIGNAS/")
    ap.add_argument("--dataset_name", required=True,
                     help="Nombre para etiquetar salidas, ej. DDI, Fitzpatrick, PADUFES, DDI2")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--internal_threshold", type=float, default=None,
                     help="Umbral de Youden congelado desde la validacion interna de BS. "
                          "Si se omite, se deriva del propio conjunto externo (menos riguroso "
                          "para un diseno verdaderamente zero-shot; documentalo si haces esto).")
    ap.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model_path, internal_mean_auc, internal_best_auc = pick_best_model(
        args.results_csv, args.model_config
    )

    print(f"\nCargando modelo congelado: {model_path}")
    model = keras.models.load_model(model_path)

    print(f"\nCargando cohorte externa '{args.dataset_name}' desde: {args.external_dir}")
    ext_df = collect_image_paths(args.external_dir, args.dataset_name)
    summarize_df(ext_df, args.dataset_name)

    ext_ds = make_dataset(ext_df, augment=False, shuffle=False)

    print("\nPrediciendo (sin reentrenar -- zero-shot)...")
    y_true, y_score = predict_scores(model, ext_ds)

    preds_path = os.path.join(args.out_dir, f"predicciones_zeroshot_{args.dataset_name}.csv")
    save_predictions(
        preds_path, y_true, y_score,
        metadata={"dataset": args.dataset_name, "model_config": args.model_config,
                  "frozen_model_file": model_path},
    )
    print(f"Predicciones guardadas: {preds_path}")

    if args.internal_threshold is not None:
        threshold = args.internal_threshold
        threshold_source = "internal_frozen_BS"
    else:
        threshold, _ = optimal_youden_threshold(y_true, y_score)
        threshold_source = f"derived_on_external_{args.dataset_name} (documentar como tal, no es zero-shot puro)"

    print(f"\nUmbral usado: {threshold:.4f}  ({threshold_source})")

    results = bootstrap_metrics(y_true, y_score, threshold, args.n_bootstrap)
    external_auc_pooled = results["auc"]["mean"] if results["n_valid_bootstrap"] else roc_auc_score(y_true, y_score)
    external_auc_pooled = roc_auc_score(y_true, y_score)  # AUC pooled real, no el promedio bootstrap

    domain_gap_vs_mean = internal_mean_auc - external_auc_pooled
    domain_gap_vs_best = internal_best_auc - external_auc_pooled

    summary = {
        "dataset": args.dataset_name,
        "model_config": args.model_config,
        "frozen_model_file": model_path,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "n_external": int(len(ext_df)),
        "internal_mean_auc_15_runs": internal_mean_auc,
        "internal_best_fold_auc": internal_best_auc,
        "external_auc_pooled": float(external_auc_pooled),
        "domain_gap_vs_internal_mean": float(domain_gap_vs_mean),
        "domain_gap_vs_internal_best": float(domain_gap_vs_best),
        "bootstrap": results,
    }

    summary_path = os.path.join(args.out_dir, f"resumen_zeroshot_{args.dataset_name}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print(f"RESULTADO ZERO-SHOT: {args.model_config} congelado -> {args.dataset_name}")
    print("=" * 65)
    print(f"AUC interno (media 15 corridas):     {internal_mean_auc:.4f}")
    print(f"AUC interno (mejor corrida, congelada): {internal_best_auc:.4f}")
    print(f"AUC externo ({args.dataset_name}, pooled):     {external_auc_pooled:.4f}  "
          f"[IC95%: {results['auc']['ci_low']:.4f}, {results['auc']['ci_high']:.4f}]")
    print(f"Sensibilidad externa: {results['sensitivity']['mean']:.4f}  "
          f"[{results['sensitivity']['ci_low']:.4f}, {results['sensitivity']['ci_high']:.4f}]")
    print(f"Especificidad externa: {results['specificity']['mean']:.4f}  "
          f"[{results['specificity']['ci_low']:.4f}, {results['specificity']['ci_high']:.4f}]")
    print(f"\nBrecha real (interno medio - externo): {domain_gap_vs_mean:.4f}")
    print(f"Brecha real (mejor corrida - externo):  {domain_gap_vs_best:.4f}")

    if domain_gap_vs_mean > 0.15:
        print("\n[interpretacion] Brecha grande: la adaptacion al sintetico NO parece "
              "transferir bien a esta cohorte real. Esto es un resultado honesto y "
              "publicable, pero contradice (o al menos matiza fuertemente) la afirmacion "
              "de 'equidad' del manuscrito original.")
    else:
        print("\n[interpretacion] Brecha moderada/pequena: hay evidencia de que la "
              "adaptacion generaliza, al menos parcialmente, a esta cohorte real.")

    # Curva ROC real
    fpr, tpr, _ = roc_curve(y_true, y_score)
    y_pred_at_thr = (y_score >= threshold).astype(int)
    tp = np.sum((y_pred_at_thr == 1) & (y_true == 1))
    fn = np.sum((y_pred_at_thr == 0) & (y_true == 1))
    fp_ = np.sum((y_pred_at_thr == 1) & (y_true == 0))
    tn = np.sum((y_pred_at_thr == 0) & (y_true == 0))
    sens_pt = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec_pt = tn / (tn + fp_) if (tn + fp_) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2,
            label=f"{args.model_config} -> {args.dataset_name} (AUC={external_auc_pooled:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC=0.500)")
    ax.scatter([1 - spec_pt], [sens_pt], color="red", zorder=5,
               label=f"Umbral usado (sens={sens_pt:.2f}, spec={spec_pt:.2f})")
    ax.set_xlabel("1 - Especificidad (FPR)")
    ax.set_ylabel("Sensibilidad (TPR)")
    ax.set_title(f"ROC zero-shot real: {args.model_config} (HAM\u2192BS) evaluado en {args.dataset_name}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    roc_path = os.path.join(args.out_dir, f"roc_zeroshot_{args.dataset_name}.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()

    print(f"\nCurva ROC guardada: {roc_path}")
    print(f"Resumen guardado: {summary_path}")


if __name__ == "__main__":
    main()
