"""Comparaciones pareadas con split por lesion: dentro de EfficientNetB0 y entre arquitecturas.
Uso: python analisis_arquitecturas.py <resultados_leakage_safe.csv> <carpeta_salida>"""
import sys, numpy as np, pandas as pd
from scipy import stats
src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src); d["arch"] = d["arch"].fillna("mobilenetv2")
d = d[~d.model.str.startswith("FINAL")].drop_duplicates(["arch", "model", "repeat", "fold"], keep="last")
cols = ["auc", "accuracy", "sensitivity", "specificity", "precision", "f1"]
s = d.groupby(["arch", "model"])[cols].mean()
s["auc_sd"] = d.groupby(["arch", "model"]).auc.std(); s["youden"] = s.sensitivity + s.specificity - 1
s["n"] = d.groupby(["arch", "model"]).size()
print(s.round(3).to_string()); s.round(4).to_csv(f"{out}/resumen_por_arquitectura.csv")
ratio = 0.25  # validacion/entrenamiento en Brown Skin (2,345 / 9,380), igual que la Tabla de comparaciones
def corrected(diff):
    k = len(diff); mu = diff.mean(); se = np.sqrt((1 / k + ratio) * diff.var(ddof=1))
    q = stats.t.ppf(0.975, k - 1); p = 2 * stats.t.sf(abs(mu / se), k - 1)
    return mu, mu - q * se, mu + q * se, p, int((diff > 0).sum())
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, min(1, (m - i) * ps[j])); adj[j] = run
    return adj
rows = []
w = d.pivot_table(index=["arch", "repeat", "fold"], columns="model", values="auc")
we = w.loc["efficientnetb0"]
for a, b in [("A", "B"), ("B", "C40"), ("C40", "D"), ("C40", "E"), ("B", "D"), ("B", "E")]:
    mu, lo, hi, p, wins = corrected((we[b] - we[a]).values)
    rows.append(dict(family="EfficientNetB0", A=a, B=b, dAUC=mu, lo=lo, hi=hi, p=p, wins=wins))
wm = w.loc["mobilenetv2"]
for m in ["A", "B", "C40", "D", "E"]:
    diff = (we[m] - wm[m]).values   # EfficientNetB0 - MobileNetV2, mismo fold
    mu, lo, hi, p, wins = corrected(diff)
    rows.append(dict(family="Eff - Mob", A=f"MobileNetV2 {m}", B=f"EfficientNetB0 {m}", dAUC=mu, lo=lo, hi=hi, p=p, wins=wins))
r = pd.DataFrame(rows)
for fam in r.family.unique():
    idx = r.family == fam; r.loc[idx, "p_holm"] = holm(r.loc[idx, "p"].values)
print(r.round(4).to_string()); r.to_csv(f"{out}/comparaciones_arquitecturas.csv", index=False)
print("ratio", round(ratio, 4))
