"""Figura ROC empirica del zero-shot (modelo FINAL_C40, MobileNetV2).
Uso: python figura_roc_zeroshot.py <carpeta_con_predicciones_y_resumenes> <salida.png>"""
import sys, json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

folder, out = sys.argv[1], sys.argv[2]
COH = [("DDI", "DDI"), ("DDI2", "DDI-2"), ("Fitzpatrick", "Fitzpatrick17k"), ("PADUFES", "PAD-UFES-20")]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.4), sharex=True, sharey=True)
for ax, (key, label) in zip(axes.ravel(), COH):
    d = pd.read_csv(os.path.join(folder, f"predicciones_zeroshot_{key}.csv"))
    j = json.load(open(os.path.join(folder, f"resumen_zeroshot_{key}.json")))
    fpr, tpr, _ = roc_curve(d.y_true, d.y_score)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.plot(fpr, tpr, color=BLUE, lw=2)
    ax.scatter([1 - j["specificity_frozen"]], [j["sensitivity_frozen"]], s=40, color=ORANGE,
               edgecolor="white", linewidth=1.5, zorder=3, label="Frozen threshold (BS validation)")
    ax.scatter([1 - j["specificity_youden"]], [j["sensitivity_youden"]], s=40, facecolor="white",
               edgecolor=BLUE, linewidth=1.8, zorder=3, label="Youden threshold (re-optimized)")
    lo, hi = j["auc_ci95"]
    ax.set_title(f"{label}  (n = {j['n']:,})", fontsize=9.5, color=INK, loc="left")
    ax.text(0.97, 0.05, f"AUC {j['auc']:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]", ha="right",
            va="bottom", transform=ax.transAxes, color=INK, fontsize=8.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
for ax in axes[1]: ax.set_xlabel("1 − Specificity")
for ax in axes[:, 0]: ax.set_ylabel("Sensitivity")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(out, dpi=300, facecolor="white")
print("ok", out)
