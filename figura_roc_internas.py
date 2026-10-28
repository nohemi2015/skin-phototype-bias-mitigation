"""ROC empiricas out-of-fold de MobileNetV2 (split por lesion).
Usa las repeticiones 2 y 3: cada repeticion cubre una vez todas las imagenes
de Brown Skin; la repeticion 1 se excluye porque dos archivos de prediccion
(A y C40, fold 1) no coinciden con los AUC registrados.
Uso: python figura_roc_internas.py <carpeta_pred_mobilenet> <carpeta_pred_efficientnet> <salida.png> <salida_auc.csv>
Los archivos de EfficientNetB0 llevan el sufijo _efficientnetb0 (pred_B_efficientnetb0_rep2_fold1.csv)."""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
folder, folder_eff, out, out_csv = sys.argv[1:5]
REPS, FOLDS = [2, 3], [1, 2, 3, 4, 5]
def load(cfg, arch="mob"):
    fd, tag = (folder, "") if arch == "mob" else (folder_eff, "_efficientnetb0")
    return pd.concat([pd.read_csv(os.path.join(fd, f"pred_{cfg}{tag}_rep{r}_fold{f}.csv"))
                      for r in REPS for f in FOLDS], ignore_index=True)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
panels = [
  ("(a) MobileNetV2: strategy", "mob", [("A", "A: HAM10000 only", "#8c2341", "-"),
                               ("B", "B: Brown Skin only", "#3aa9b1", "--"),
                               ("C40", "C40: HAM + FT (40)", "#006d74", "-"),
                               ("D", "D: HAM ∪ BS", "#3aa9b1", ":"),
                               ("E", "E: HAM ∪ BS balanced", "#006d74", "-.")]),
  ("(b) MobileNetV2: depth", "mob", [("C10", "C10", "#9ad3d7", "-"), ("C20", "C20", "#5bbcc3", "-"),
                             ("C40", "C40", "#00939c", "-"), ("C60", "C60", "#00737b", "-"),
                             ("C80", "C80", "#004d55", "-")]),
  ("(c) EfficientNetB0: strategy", "eff", [("A", "A: HAM10000 only", "#8c2341", "-"),
                               ("B", "B: Brown Skin only", "#3aa9b1", "--"),
                               ("C40", "C40: HAM + FT (40)", "#006d74", "-"),
                               ("D", "D: HAM ∪ BS", "#3aa9b1", ":"),
                               ("E", "E: HAM ∪ BS balanced", "#006d74", "-.")]),
]
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.95), sharey=True)
rows = []
for ax, (title, arch, cfgs) in zip(axes, panels):
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
    for cfg, label, color, ls in cfgs:
        d = load(cfg, arch); fpr, tpr, _ = roc_curve(d.y_true, d.y_score)
        auc = roc_auc_score(d.y_true, d.y_score)
        ax.plot(fpr, tpr, color=color, lw=1.8, ls=ls, label=f"{label} ({auc:.3f})")
        key = f"{arch}_{cfg}"
        if key not in [r["config"] for r in rows]:
            rows.append({"config": key, "pooled_oof_auc_rep2_3": round(auc, 4), "n": len(d)})
    ax.set_title(title, loc="left", fontsize=8, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("1 − Specificity", color=INK)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=5.6, frameon=True, framealpha=0.95, edgecolor=GRID,
              title="Pooled OOF AUC", title_fontsize=5.8, handlelength=1.8, borderpad=0.4, labelspacing=0.3)
axes[0].set_ylabel("Sensitivity", color=INK)
fig.tight_layout(); fig.savefig(out, dpi=300, facecolor="white")
pd.DataFrame(rows).to_csv(out_csv, index=False); print(pd.DataFrame(rows))
