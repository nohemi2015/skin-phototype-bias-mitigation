"""Resumen gráfico (graphical abstract) para Computers in Biology and Medicine.
Uso: python figura_resumen_grafico.py auc_fase2.csv salida   (escribe salida.pdf, salida.tiff, salida.png)

Valores internos: medias de la validación cruzada por lesión (Tabla 7).
Valores externos: AUC zero-shot de A y de C40 (media de 3 semillas) leídos de analisis_fase2.py;
si existe A_mean (A con 3 semillas) se usa en lugar de A.
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
r = pd.read_csv(sys.argv[1]); out = sys.argv[2]

TEAL, WINE, INK, MUTED, GRID, BG = "#00939c", "#8c2341", "#1f1f1e", "#6b6a64", "#e4e3dd", "#f3f2ee"
INTERNAL = {"mobilenetv2": (0.609, 0.772), "efficientnetb0": (0.675, 0.775)}
COLORS = {"mobilenetv2": TEAL, "efficientnetb0": WINE}
NAMES = {"mobilenetv2": "MobileNetV2", "efficientnetb0": "EfficientNet-B0"}
COH = ["DDI", "DDI2", "Fitzpatrick", "PADUFES"]
a_key = "A_mean" if (r.model == "A_mean").any() else "A"

def auc(arch, co, m):
    return float(r[(r.arch == arch) & (r.cohort == co) & (r.model == m)].auc.iloc[0])

ext = {arch: [(auc(arch, c, a_key), auc(arch, c, "C40_mean")) for c in COH] for arch in COLORS}
wins = sum(a > c for arch in ext for a, c in ext[arch])
c40_all = [c for arch in ext for _, c in ext[arch]]

fig = plt.figure(figsize=(10, 4.3))
fig.patch.set_facecolor("white")
YL = (0.45, 0.85)

def title(x, num, text):
    fig.text(x, 0.93, num, fontsize=13, fontweight="bold", color="white", ha="center", va="center",
             bbox=dict(boxstyle="circle,pad=0.25", fc=INK, ec="none"))
    fig.text(x + 0.022, 0.93, text, fontsize=12.5, fontweight="bold", color=INK, va="center")

# ---------- Panel 1: datos y partición ----------
ax = fig.add_axes([0.01, 0.17, 0.30, 0.68]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
title(0.025, "1", "Synthetic target")

def skin(x, y, tone, label, sub):
    ax.add_patch(FancyBboxPatch((x - 1.9, y - 2.9), 3.8, 5.4, boxstyle="round,pad=0.1,rounding_size=0.4",
                                fc=BG, ec="none"))
    ax.add_patch(Circle((x, y + 1.15), 1.05, fc=tone, ec="none"))
    ax.add_patch(Ellipse((x + 0.1, y + 1.1), 0.85, 0.65, angle=20, fc="#4a2c22", ec="none", alpha=0.85))
    ax.text(x, y - 0.45, label, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(x, y - 1.05, sub, ha="center", va="top", fontsize=9.5, color=MUTED, linespacing=1.15)

skin(2.1, 6.0, "#f1d3bd", "HAM10000", "real dermoscopy\nmostly lighter skin")
skin(7.9, 6.0, "#8d5a3b", "Brown Skin", "synthetic\nFST IV–VI")
ax.add_patch(FancyArrowPatch((4.15, 6.9), (5.85, 6.9), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK))
ax.text(5.0, 7.35, "style\ntransfer", ha="center", va="bottom", fontsize=9, color=MUTED, linespacing=1.0)
ax.text(5.0, 1.2, "Lesion-level split:\na lesion and its synthetic copies\nstay in the same fold",
        ha="center", va="center", fontsize=9.8, color=INK, linespacing=1.15)

# ---------- Panel 2: evaluación interna ----------
ax2 = fig.add_axes([0.39, 0.22, 0.22, 0.60])
title(0.345, "2", "Internal (synthetic)")
xs = {"mobilenetv2": 0, "efficientnetb0": 1}
for arch, (a, c) in INTERNAL.items():
    x = xs[arch]
    ax2.bar(x - 0.19, a, 0.34, color="#bdbcb5")
    ax2.bar(x + 0.19, c, 0.34, color=COLORS[arch])
    ax2.text(x - 0.19, a + 0.008, f"{a:.2f}", ha="center", fontsize=9.5, color=MUTED)
    ax2.text(x - 0.19, 0.462, "HAM\nonly", ha="center", va="bottom", fontsize=8.5, color=INK, linespacing=1.0)
    ax2.text(x + 0.19, 0.462, "C40", ha="center", va="bottom", fontsize=8.5, color="white", fontweight="bold")
    ax2.text(x + 0.19, c + 0.008, f"{c:.2f}", ha="center", fontsize=9.5, color=COLORS[arch], fontweight="bold")
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["MobileNetV2", "EfficientNet-B0"], fontsize=9.5)
ax2.set_ylim(*YL); ax2.set_ylabel("AUC", fontsize=10.5, color=INK)
ax2.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=0)
ax2.text(0.5, 0.835, "C40 fine-tuning: internal gain", ha="center", fontsize=9.5, color=INK, va="top")

# ---------- Panel 3: zero-shot en cohortes reales ----------
ax3 = fig.add_axes([0.71, 0.22, 0.27, 0.60])
title(0.655, "3", "Zero-shot, 4 real cohorts")
for arch, pairs in ext.items():
    for a, c in pairs:
        ax3.plot([0, 1], [a, c], color=COLORS[arch], lw=1.6, alpha=0.9, marker="o", ms=5.5, mec="white", mew=0.8)
ax3.set_xticks([0, 1]); ax3.set_xticklabels(["HAM10000\nonly", "C40\nadapted"], fontsize=9.5)
ax3.set_xlim(-0.35, 1.35); ax3.set_ylim(*YL); ax3.set_yticklabels([])
ax3.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
ax3.text(0.5, 0.835, f"HAM10000-only better in {wins} of 8", ha="center", fontsize=9.5, color=INK, va="top",
         fontweight="bold")
ax3.text(1.33, 0.465, f"C40, mean of 3 seeds: {min(c40_all):.2f}–{max(c40_all):.2f}", ha="right", fontsize=9, color=MUTED)
for arch, y in (("mobilenetv2", 0.80), ("efficientnetb0", 0.775)):
    ax3.plot([-0.3, -0.18], [y, y], color=COLORS[arch], lw=2)
    ax3.text(-0.14, y, NAMES[arch], fontsize=8.5, color=COLORS[arch], va="center")

for a in (ax2, ax3):
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.spines["left"].set_color(MUTED); a.spines["bottom"].set_color(MUTED)
    a.tick_params(colors=INK, labelsize=9.5); a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)

fig.text(0.5, 0.045, "Internal gains on a synthetic dark-skin target did not transfer to real clinical photographs",
         ha="center", fontsize=12, color="white", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.45", fc=INK, ec="none"))

fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=200)
fig.savefig(out + ".tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
print("ok", out, "wins", wins)
