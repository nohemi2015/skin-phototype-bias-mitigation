"""Figura: AUC zero-shot de A, B y C40 (media de 3 semillas) por cohorte, ambos backbones.
Uso: python figura_A_vs_C40.py auc_fase2.csv out.png"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = pd.read_csv(sys.argv[1]); out = sys.argv[2]
TEAL, WINE, INK, MUTED, GRID = "#00939c", "#8c2341", "#1f1f1e", "#6b6a64", "#e4e3dd"
A_COL, B_COL = "#2f2f2d", "#a3a29c"
control = {"mobilenetv2": {"A": 0.613, "B": 0.764, "C40_mean": 0.771},
           "efficientnetb0": {"A": 0.672, "B": 0.791, "C40_mean": 0.705}}
coh = ["DDI", "DDI2", "Fitzpatrick", "PADUFES"]
lab = ["Brown Skin\ncontrol\n(synthetic)", "DDI", "DDI-2", "Fitz-\npatrick17k\n(FST IV–VI)", "PAD-\nUFES-20"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
models = [("A", "A: HAM10000 only", A_COL, "o"), ("B", "B: Brown Skin only", B_COL, "s"),
          ("C40_mean", "C40 (mean of 3 seeds)", None, "^")]
off = {"A": -0.22, "B": 0.0, "C40_mean": 0.22}
for ax, arch, name, col in [(axes[0], "mobilenetv2", "MobileNetV2", TEAL), (axes[1], "efficientnetb0", "EfficientNetB0", WINE)]:
    ax.axvspan(-0.5, 0.5, color="#f3f2ee", zorder=0)
    for key, lbl, c, mk in models:
        c = c or col
        ys = [control[arch][key]]; lo = [np.nan]; hi = [np.nan]
        for co in coh:
            row = r[(r.arch == arch) & (r.cohort == co) & (r.model == key)].iloc[0]
            ys.append(row.auc); lo.append(row.lo); hi.append(row.hi)
        x = np.arange(5) + off[key]; ys, lo, hi = map(np.array, (ys, lo, hi))
        ax.vlines(x[1:], lo[1:], hi[1:], color=c, lw=1.4, zorder=2)
        ax.plot(x, ys, ls="none", marker=mk, ms=6.5, color=c, mec="white", mew=1.2, zorder=3, label=lbl)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=7.6)
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(0.45, 0.85)
    ax.set_title(name, fontsize=9.5, color=INK, loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("AUC-ROC", color=INK)
from matplotlib.lines import Line2D
h = [Line2D([], [], ls="none", marker="o", ms=6.5, color=A_COL, mec="white", label="A: HAM10000 only"),
     Line2D([], [], ls="none", marker="s", ms=6.5, color=B_COL, mec="white", label="B: Brown Skin only"),
     Line2D([], [], ls="none", marker="^", ms=6.5, color=TEAL, mec="white", label="C40, MobileNetV2"),
     Line2D([], [], ls="none", marker="^", ms=6.5, color=WINE, mec="white", label="C40, EfficientNetB0"),
     Line2D([], [], color=MUTED, lw=1, ls=(0, (3, 3)), label="Chance")]
fig.legend(handles=h, frameon=False, fontsize=7.8, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2)
fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.93)); fig.savefig(out, dpi=300, facecolor="white"); print("ok", out)
