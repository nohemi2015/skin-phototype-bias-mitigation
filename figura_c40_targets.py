"""Figura 3: AUC de C40 segun el dominio objetivo del fine-tuning."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1]
groups = ["Source only (A)\nevaluated on Brown Skin", "C40 → Brown Skin\n(synthetic)", "C40 → Fitzpatrick17k\n(real, FST IV–VI)", "C40 → PAD-UFES-20\n(real)"]
mob = [0.610, 0.772, 0.825, 0.845]
eff = [0.675, 0.775, 0.807, 0.864]
BLUE, ORANGE, INK, MUTED, GRID = "#00939c", "#8c2341", "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(groups)); w = 0.36
b1 = ax.bar(x - w/2 - 0.01, mob, w, color=BLUE, label="MobileNetV2", edgecolor="white", linewidth=1)
b2 = ax.bar(x + w/2 + 0.01, eff, w, color=ORANGE, label="EfficientNetB0", edgecolor="white", linewidth=1)
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.012, f"{r.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5, color=INK)
ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), label="Chance (AUC = 0.5)")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylim(0, 1); ax.set_ylabel("Internal AUC-ROC", color=INK)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8.5)
fig.tight_layout(); fig.savefig(out, dpi=300, facecolor="white"); print("ok", out)
