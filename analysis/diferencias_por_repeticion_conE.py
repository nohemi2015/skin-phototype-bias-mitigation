"""Sensibilidad de las comparaciones pareadas de MobileNetV2: diferencia media de AUC por
repeticion (cada repeticion = una CV completa de 5 folds) y rango de las 15 diferencias.
Uso: python diferencias_por_repeticion.py <fold_level_results.csv> <carpeta_salida>"""
import sys, pandas as pd
src, out = sys.argv[1], sys.argv[2]
r = pd.read_csv(src)
if "arch" in r: r = r[r["arch"].fillna("mobilenetv2") == "mobilenetv2"]
r = r[~r.model.astype(str).str.startswith("FINAL")].drop_duplicates(["model", "repeat", "fold"], keep="last")
w = r.pivot_table(index=["repeat", "fold"], columns="model", values="auc")
comps = [("A", "B"), ("B", "C10"), ("B", "C20"), ("B", "C40"), ("B", "C60"), ("B", "C80"),
         ("C20", "C40"), ("C40", "D"), ("C40", "E"), ("B", "D"), ("B", "E")]
rows = []
for a, b in comps:
    diff = w[b] - w[a]; rep = diff.groupby(level=0).mean()
    rows.append({"A": a, "B": b, "rep1": rep[1], "rep2": rep[2], "rep3": rep[3],
                 "min_fold": diff.min(), "median_fold": diff.median(), "max_fold": diff.max(),
                 "positive_folds": int((diff > 0).sum())})
t = pd.DataFrame(rows); print(t.round(4).to_string())
t.to_csv(f"{out}/diferencias_por_repeticion.csv", index=False)
