"""Figura 1: diagrama de la metodologia (vectorial).
Uso: python figura_metodologia.py <salida_sin_extension>  -> .pdf y .png (300 dpi)"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse

out = sys.argv[1]
INK, MUTED, LINE = "#1f1f1e", "#5f5e58", "#9a998f"
BLUE, BLUE_BG = "#2a78d6", "#e8f1fc"
ORANGE, ORANGE_BG = "#c9541f", "#fdeee7"
GREY_BG = "#f3f2ee"
plt.rcParams.update({"font.family": "DejaVu Sans"})
FW, FH = 7.2, 4.0
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100 * FH / FW); ax.axis("off")
TOP = 100 * FH / FW - 1.0
T, B = 7.0, 6.2                    # tamaño de titulo y de cuerpo (pt)
LT, LB = 1.95, 1.62                # alto de linea en unidades
PAD = 1.1

def rbox(x, y, w, h, fc, ec, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=lw))

def block_height(sections):
    h = PAD
    for i, (title, body) in enumerate(sections):
        if title: h += LT
        h += LB * len(body)
        if i < len(sections) - 1: h += 0.9
    return h + PAD - 0.2

def block(x, ytop, w, sections, fc="white", ec=INK, lw=0.9, title_color=INK, badge=None,
          badge_fc=INK, badge_tc="white"):
    h = block_height(sections)
    rbox(x, ytop - h, w, h, fc, ec, lw)
    tx = x + 1.1 + (4.0 if badge else 0)
    if badge:
        ax.add_patch(FancyBboxPatch((x + 1.0, ytop - PAD - 2.6), 2.8, 2.6,
                                    boxstyle="round,pad=0,rounding_size=0.5", fc=badge_fc, ec=badge_fc))
        ax.text(x + 2.4, ytop - PAD - 1.3, badge, fontsize=7.2, fontweight="bold", color=badge_tc,
                ha="center", va="center")
    y = ytop - PAD
    for i, (title, body) in enumerate(sections):
        if title:
            ax.text(tx, y - LT / 2, title, fontsize=T, fontweight="bold", color=title_color, va="center")
            y -= LT
        for line in body:
            bold = line.startswith("**")
            ax.text(tx, y - LB / 2, line.strip("*"), fontsize=B, color=INK if bold else MUTED,
                    fontweight="bold" if bold else "normal", va="center")
            y -= LB
        if i < len(sections) - 1: y -= 0.9
    return ytop - h

def cylinder(cx, ytop, w=3.0, h=3.4, fc="#ffffff", ec=INK, n=3):
    """Icono de base de datos: n discos apilados."""
    dh = h / n; eh = w * 0.38
    for i in range(n):
        yb = ytop - eh / 2 - (i + 1) * dh
        ax.add_patch(Rectangle((cx - w / 2, yb), w, dh, fc=fc, ec="none", zorder=3))
        ax.plot([cx - w / 2] * 2, [yb, yb + dh], color=ec, lw=0.7, zorder=4)
        ax.plot([cx + w / 2] * 2, [yb, yb + dh], color=ec, lw=0.7, zorder=4)
        ax.add_patch(Ellipse((cx, yb), w, eh, fc=fc, ec=ec, lw=0.7, zorder=4))
    ax.add_patch(Ellipse((cx, ytop - eh / 2), w, eh, fc=fc, ec=ec, lw=0.7, zorder=5))

def fold_grid(x, ytop, cell=0.85, gap=0.25, color=INK):
    """Icono de validacion cruzada: 3 repeticiones x 5 folds, un fold de validacion por fila."""
    for r in range(3):
        for c in range(5):
            val = (c == (r * 2) % 5)
            ax.add_patch(Rectangle((x + c * (cell + gap), ytop - (r + 1) * (cell + gap)), cell, cell,
                                   fc=color if val else "white", ec=color, lw=0.6, zorder=3))

def depth_scale(x, y, w, color):
    """Mini escala de capas descongeladas (10-80 de las ultimas capas)."""
    ax.plot([x, x + w], [y, y], color=color, lw=1.6, solid_capstyle="butt")
    for i, v in enumerate((10, 20, 40, 60, 80)):
        xx = x + w * i / 4
        ax.plot([xx, xx], [y - 0.45, y + 0.45], color=color, lw=0.8)
        ax.text(xx, y - 0.75, str(v), fontsize=5.2, color=MUTED, ha="center", va="top")
    ax.text(x + w + 0.8, y, "layers", fontsize=5.4, color=MUTED, va="center")

def arrow(x0, y0, x1, y1, color=LINE, ls="-", lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))

def note(x, y, s, ha="left", color=MUTED):
    ax.text(x, y, s, fontsize=5.8, color=color, ha=ha, va="center")

M, W, G = 1.2, 22.8, 2.4
X = [M + i * (W + G) for i in range(4)]
for x, s in zip(X, ["1  Training data and partition", "2  Backbones and strategies",
                    "3  Internal evaluation", "4  External zero-shot test"]):
    ax.text(x, TOP - 0.8, s, fontsize=7.6, fontweight="bold", color=MUTED, va="center")
Y0 = TOP - 3.2

# ---------- columna 1 ----------
x = X[0]
y = block(x, Y0, W, [("HAM10000  (source)", ["9,672 dermoscopic images", "mostly Fitzpatrick I–III",
                                             "7,307 lesions"])], fc=BLUE_BG, ec=BLUE)
cylinder(x + W - 3.0, y + 4.9, fc="#f6dfc8", ec="#8a6a4f")
arrow(x + 4, y, x + 4, y - 2.6, color=BLUE, ls=(0, (2, 1.4)))
note(x + 5.2, y - 1.3, "style transfer", color=BLUE)
y = block(x, y - 2.6, W, [("Brown Skin  (synthetic)", ["11,725 images:",
                                                              "HAM10000 lesions recoloured",
                                                              "to Fitzpatrick IV–VI",
                                                              "(VGG19 style transfer)",
                                                              "3,133 parent lesions"])], fc=BLUE_BG, ec=BLUE)
cylinder(x + W - 3.0, y + 4.9, fc="#8d5b3d", ec="#4a2e1d")
arrow(x + W / 2, y, x + W / 2, y - 2.6)
y_part_top = y - 2.6
y = block(x, y_part_top, W, [("Lesion-level partition", ["both datasets pooled and grouped",
                                                         "by HAM10000 lesion_id",
                                                         "(7,470 groups): a lesion and all",
                                                         "its synthetic derivatives always",
                                                         "fall in the same fold"])], lw=1.2)
y_part_mid = (y_part_top + y) / 2
block(x, y - 2.4, W, [("Preprocessing", ["resize to 224 × 224, scale to [0, 1]",
                                         "training only: flips, brightness,", "contrast"])],
      fc=GREY_BG, ec=LINE)

# ---------- columna 2 ----------
x = X[1]
yb = block(x, Y0, W, [("Backbones", ["MobileNetV2 (3.4 M params)", "EfficientNetB0 (5.3 M params)",
                                     "ImageNet-pretrained, same head"])],
           fc=GREY_BG, ec=LINE)
y = yb - 2.4
strategies = [("A", "Source-only", ["train on HAM10000"], "#d95926", "#fdeee7", "white"),
              ("B", "Target-only", ["train on Brown Skin"], "#199e70", "#e6f5ef", "white"),
              ("C", "Transfer  C10–C80", ["start from A; unfreeze", "the last n layers on BS", ""],
               "#1c5cab", "#e8f1fc", "white"),
              ("D", "Fusion", ["HAM10000 ∪ Brown Skin"], "#c98500", "#fdf4dc", "white"),
              ("E", "Balanced fusion", ["D, Brown Skin oversampled"], "#7b4fbf", "#f1ebfa", "white")]
tops = []
for k, name, body, col, tint, tc in strategies:
    tops.append(y)
    yb_ = block(x, y, W, [(name, body)], fc=tint, ec=col, lw=1.3 if k == "C" else 0.9, badge=k,
                badge_fc=col, badge_tc=tc)
    if k == "C":
        depth_scale(x + 5.2, y - PAD - LT - 2.45 * LB, 11.5, col)
    y = yb_ - 1.0
strat_mid = (tops[0] + y) / 2
arrow(X[0] + W, y_part_mid, x, y_part_mid)

# ---------- columna 3 ----------
x = X[2]
y_cv_top = Y0
y = block(x, Y0, W, [("Stratified group CV", ["5 folds × 3 repeats, identical", "folds for every strategy,",
                                              "validation on Brown Skin"]),
                     ("Metrics", ["AUC (primary); sensitivity,", "specificity, precision, F1,",
                                  "Youden index; source–target gap"]),
                     ("Paired comparisons", ["corrected repeated-CV", "t-test, Holm-adjusted;", "both backbones"])], lw=1.2)
y_cv_mid = (y_cv_top + y) / 2
fold_grid(x + W - 6.6, y + 4.6)
ya = tops[1] - 2.8
arrow(X[1] + W, ya, x, ya)
block(x, y - 2.4, W, [("Image-level partition", ["original random split and earlier",
                                                 "pipeline, both backbones; kept",
                                                 "only as a sensitivity analysis"])], fc=GREY_BG, ec=LINE)

# ---------- columna 4 ----------
x = X[3]
y = block(x, Y0, W, [("Final models", ["A, B and C40, both backbones, on", "one lesion-level development",
                                          "split (85 %), no fold selection;", "C40 with 3 seeds; threshold",
                                          "frozen on Brown Skin validation"])], lw=1.2)
arrow(X[2] + W, Y0 - 5.5, x, Y0 - 5.5)
pass
arrow(x + W / 2, y, x + W / 2, y - 2.6, color=INK)
note(x + W / 2 + 1.0, y - 1.3, "no retraining")
cylinder(x + W - 3.0, y - 2.6 - 1.1, fc="#f7c9b2", ec=ORANGE)
block(x, y - 2.6, W, [("Real external cohorts", ["clinical photographs,", "never used for training"]),
                      ("", ["**DDI", "656 images, FST I–VI",
                            "**DDI-2", "661 images, self-identified Asian,", "mostly FST III–IV",
                            "**Fitzpatrick17k", "1,079 images, FST IV–VI",
                            "**PAD-UFES-20", "2,298 smartphone images"]),
                      ("", ["control: Brown Skin hold-out",
                            "AUC with bootstrap 95 % CI;",
                            "frozen vs. re-optimized threshold;", "paired A vs. C40 comparison"])],
      fc=ORANGE_BG, ec=ORANGE, title_color=ORANGE)

fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300, facecolor="white")
print("ok")
