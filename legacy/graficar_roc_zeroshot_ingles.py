"""
graficar_roc_zeroshot_ingles.py

Regenera las curvas ROC del experimento zero-shot (Strand 3) EN INGLES,
a partir de los archivos ya generados por evaluar_zeroshot.py:
    predicciones_zeroshot_<dataset>.csv   (columnas y_true, y_score)
    resumen_zeroshot_<dataset>.json       (para el umbral usado)

No vuelve a cargar ningun modelo ni a predecir nada -- solo relee los
resultados ya calculados y rehace la grafica con etiquetas en ingles.
Genera tanto las 5 figuras individuales como una cuadricula combinada
(2 filas x 3 columnas) lista para el manuscrito.

Uso:
    python graficar_roc_zeroshot_ingles.py ^
        --entrada "BrownSkin=resultados\\zeroshot_BrownSkin,DDI=resultados\\zeroshot_DDI,DDI2=resultados\\zeroshot_DDI2,Fitzpatrick17k=resultados\\zeroshot_Fitzpatrick17k,PADUFES=resultados\\zeroshot_PADUFES" ^
        --out_dir roc_zeroshot_en

Cada entrada es "Etiqueta=carpeta_de_resultados_de_evaluar_zeroshot".
La etiqueta se usa tal cual en el titulo/leyenda de la grafica.
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
from PIL import Image

# Nombre "bonito" para mostrar en la grafica (ajusta si tus carpetas
# usan otras etiquetas)
DISPLAY_NAME = {
    "BrownSkin": "Brown Skin (control)",
    "DDI": "DDI (real)",
    "DDI2": "DDI-2 (real)",
    "Fitzpatrick17k": "Fitzpatrick17k (real)",
    "PADUFES": "PAD-UFES-20 (real)",
}


def find_files(folder, label):
    """Busca el CSV de predicciones y el JSON de resumen dentro de la
    carpeta, sin asumir un nombre exacto (por si quedaron renombrados)."""
    pred_csv, summary_json = None, None
    for fname in os.listdir(folder):
        low = fname.lower()
        if low.startswith("predicciones") and low.endswith(".csv"):
            pred_csv = os.path.join(folder, fname)
        if low.startswith("resumen") and low.endswith(".json"):
            summary_json = os.path.join(folder, fname)
    if pred_csv is None:
        raise FileNotFoundError(f"No se encontro un CSV de predicciones en {folder}")
    return pred_csv, summary_json


def plot_one(label, folder, out_dir):
    pred_csv, summary_json = find_files(folder, label)
    df = pd.read_csv(pred_csv)
    y_true, y_score = df["y_true"].values, df["y_score"].values

    auc_val = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    threshold = None
    if summary_json and os.path.exists(summary_json):
        with open(summary_json) as f:
            summary = json.load(f)
        threshold = summary.get("threshold")

    if threshold is not None:
        y_pred = (y_score >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        fp_ = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp_) if (tn + fp_) > 0 else np.nan
    else:
        sens = spec = None

    display_name = DISPLAY_NAME.get(label, label)

    fig, ax = plt.subplots(figsize=(5, 5.4))
    ax.plot(fpr, tpr, linewidth=2, color="#1f77b4",
            label=f"C40 -> {display_name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            label="Chance (AUC=0.500)")
    if sens is not None:
        ax.scatter([1 - spec], [sens], color="red", zorder=5,
                   label=f"Threshold used (sens={sens:.2f}, spec={spec:.2f})")

    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(f"Zero-shot ROC: C40 (HAM\u2192BS)\nevaluated on {display_name}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"roc_zeroshot_{label}_en.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"{label}: AUC={auc_val:.4f}  guardado en {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True,
                     help="Lista 'Etiqueta=carpeta' separadas por coma")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--grid_out", default=None,
                     help="Nombre del archivo de cuadricula combinada "
                          "(default: <out_dir>/roc_zeroshot_grid_en.png)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pairs = [p for p in args.entrada.split(",") if p.strip()]
    generated = []
    for pair in pairs:
        if "=" not in pair:
            print(f"[aviso] entrada mal formada, se omite: {pair}")
            continue
        label, folder = pair.split("=", 1)
        label, folder = label.strip(), folder.strip()
        try:
            path = plot_one(label, folder, args.out_dir)
            generated.append(path)
        except FileNotFoundError as e:
            print(f"[aviso] {label}: {e}")

    if not generated:
        print("No se genero ninguna grafica.")
        return

    # Cuadricula combinada (hasta 3 columnas)
    imgs = [Image.open(p).convert("RGB") for p in generated]
    w, h = imgs[0].size
    imgs = [im.resize((w, h)) for im in imgs]

    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    grid = Image.new("RGB", (w * cols, h * rows), "white")
    for i, im in enumerate(imgs):
        cx, cy = i % cols, i // cols
        grid.paste(im, (cx * w, cy * h))

    grid_out = args.grid_out or os.path.join(args.out_dir, "roc_zeroshot_grid_en.png")
    grid.save(grid_out)
    print(f"\nCuadricula combinada guardada en: {grid_out}")


if __name__ == "__main__":
    main()
