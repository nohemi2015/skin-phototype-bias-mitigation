"""Figura: (a) AUC zero-shot en Derm7pt, misma lesion en dermatoscopia y en foto clinica,
para A, B y C40 (media de 3 semillas, IC 95 % bootstrap por caso); (b) distribucion del
ITA (tono de piel) por base.
Uso: python figura_derm7pt_ita.py auc_derm7pt.csv ita_usadas.csv salida.png"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
r = pd.read_csv(sys.argv[1]); ita = pd.read_csv(sys.argv[2]); out = sys.argv[3]
TEAL, WINE, INK, MUTED, GRID = "#00939c", "#8c2341", "#1f1f1e", "#6b6a64", "#e4e3dd"
COL = {"A": "#2f2f2d", "B": "#a3a29c", "C40": None, "D": "#c98500", "E": "#e3b75c"}
MK = {"A": "o", "B": "s", "C40": "^", "D": "D", "E": "v"}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.5), gridspec_kw={"width_ratios": [1.2, 1.25]})

# (a) Derm7pt: por backbone, A/B/C40; relleno = dermatoscopia, hueco = foto clinica
for k, (arch, ac, name) in enumerate((("mobilenetv2", TEAL, "MobileNetV2"), ("efficientnetb0", WINE, "EfficientNet-B0"))):
    cfgs = [c for c in ("A", "B", "C40", "D") if ((r.arch == arch) & (r.cfg == c)).any()]
    for i, cfg in enumerate(cfgs):
        c = COL[cfg] or ac
        x = k * 4.6 + i
        pts = []
        for dx, mod, fill in ((-0.15, "dermoscopic", True), (0.15, "clinical", False)):
            row = r[(r.arch == arch) & (r.cfg == cfg) & (r.modality == mod)].iloc[0]
            ax.vlines(x + dx, row.lo, row.hi, color=c, lw=1.2)
            ax.plot(x + dx, row.auc, MK[cfg], ms=6.5, color=c, mec=c, mew=1.3, mfc=c if fill else "white", zorder=3)
            pts.append((x + dx, row.auc))
        ax.plot(*zip(*pts), color=c, lw=0.9, alpha=0.7)
    ax.text(k * 4.6 + 1.5, 0.865, name, ha="center", color=ac, fontweight="bold")
ax.set_xticks([k * 4.6 + i for k in (0, 1) for i in range(4)]); ax.set_xticklabels(["A", "B", "C40", "D"] * 2, fontsize=8)
ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.set_ylim(0.40, 0.88); ax.set_xlim(-0.6, 8.2); ax.set_ylabel("AUC-ROC on Derm7pt (same lesions)")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], ls="none", marker="o", color=MUTED, mec=MUTED, label="dermoscopic image"),
                   Line2D([], [], ls="none", marker="o", color=MUTED, mec=MUTED, mfc="white", mew=1.3, label="clinical image")],
          frameon=False, fontsize=7.5, loc="lower right", ncol=2, handletextpad=0.2, columnspacing=0.8)
ax.set_title("(a) Modality control (Derm7pt)", loc="left", fontsize=9.5, fontweight="bold")

# (b) ITA
groups = [("HAM10000", "HAM10000", None, "#c9c7bf"),
          ("Brown Skin IV", "BrownSkin", "IV", "#d9b38c"),
          ("Brown Skin V", "BrownSkin", "V", "#b07a4f"),
          ("Brown Skin VI", "BrownSkin", "VI", "#5a3a26"),
          ("DDI I–II", "DDI", "12", "#9ec9cc"),
          ("DDI III–IV", "DDI", "34", "#4fa3a8"),
          ("DDI V–VI", "DDI", "56", "#1d6f74"),
          ("Fitzpatrick17k IV–VI", "Fitzpatrick", None, "#7a6a9a"),
          ("DDI-2", "DDI2", None, "#b8b6ae"),
          ("PAD-UFES-20", "PADUFES", None, "#b8b6ae")]
data, labels, cols = [], [], []
for lab, base, g, c in groups:
    s = ita[ita.base == base]
    if g is not None:
        s = s[s.grupo.astype(str).str.replace(r"\.0$", "", regex=True) == g]
    data.append(s.ita.clip(-100, 130).values); labels.append(lab); cols.append(c)
pos = np.arange(len(data))[::-1]
bp = bx.boxplot(data, positions=pos, orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True,
                medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_edgecolor(MUTED)
for t in (55, 41, 28, 10, -30):
    bx.axvline(t, color=GRID, lw=0.8, zorder=0)
for t, lab in ((92, "very light"), (48, "light"), (34.5, "int."), (19, "tan"), (-10, "brown"), (-65, "dark")):
    bx.text(t, -1.05, lab, ha="center", fontsize=6.5, color=MUTED)
bx.set_ylim(-1.4, len(data) - 0.4)
bx.set_yticks(pos); bx.set_yticklabels(labels, fontsize=7.8)
bx.set_xlabel("Individual typology angle, ITA (°)"); bx.set_xlim(-100, 130)
bx.set_title("(b) Skin tone of the surrounding skin", loc="left", fontsize=9.5, fontweight="bold")
for a in (ax, bx):
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.spines["left"].set_color(MUTED); a.spines["bottom"].set_color(MUTED)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.5)
fig.savefig(out, dpi=300, facecolor="white"); print("ok", out)
