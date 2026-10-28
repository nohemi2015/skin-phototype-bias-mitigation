"""Figura ROC zero-shot con los dos modelos finales C40.
Uso: python figura_roc_zeroshot.py <carpeta_mobilenet> <carpeta_efficientnet> <salida.png>"""
from pathlib import Path
def _f(folder, name):
    """Busca el archivo en la carpeta o en sus subcarpetas (estructura plana o <cohorte>/)."""
    hits = sorted(Path(folder).rglob(name))
    if not hits: raise FileNotFoundError(f"{name} no esta en {folder}")
    return hits[0]

import sys, os, json, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
fm, fe, out = sys.argv[1:4]
COH = [("DDI", "DDI"), ("DDI2", "DDI-2"), ("Fitzpatrick", "Fitzpatrick17k (FST IV–VI)"), ("PADUFES", "PAD-UFES-20")]
MODELS = [(fm, "MobileNetV2", "#00939c", "-"), (fe, "EfficientNetB0", "#8c2341", "--")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.5), sharex=True, sharey=True)
for ax, (key, label) in zip(axes.ravel(), COH):
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
    n = None
    for folder, name, color, ls in MODELS:
        d = pd.read_csv(_f(folder, f"predicciones_zeroshot_{key}.csv"))
        j = json.load(open(_f(folder, f"resumen_zeroshot_{key}.json")))
        fpr, tpr, _ = roc_curve(d.y_true, d.y_score); n = len(d)
        ax.plot(fpr, tpr, color=color, lw=1.8, ls=ls, label=f"{name} ({roc_auc_score(d.y_true, d.y_score):.3f})")
        ax.scatter([1 - j["specificity_frozen"]], [j["sensitivity_frozen"]], s=34, color=color,
                   edgecolor="white", linewidth=1.3, zorder=3)
    ax.set_title(f"{label}  (n = {n:,})", fontsize=9, color=INK, loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.95, edgecolor=GRID, title="AUC", title_fontsize=7)
for ax in axes[1]: ax.set_xlabel("1 − Specificity", color=INK)
for ax in axes[:, 0]: ax.set_ylabel("Sensitivity", color=INK)
fig.text(0.5, 0.005, "Filled markers: operating point at each model's frozen threshold (strict zero-shot)",
         ha="center", va="bottom", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out, dpi=300, facecolor="white"); print("ok", out)
