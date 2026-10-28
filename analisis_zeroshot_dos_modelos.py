"""Zero-shot de los dos modelos finales: IC (PAD por paciente) y diferencia pareada Eff - Mob por cohorte.
Uso: python analisis_zeroshot_dos_modelos.py <carpeta_mobilenet> <carpeta_efficientnet> <salida.csv>"""
from pathlib import Path
def _f(folder, name):
    """Busca el archivo en la carpeta o en sus subcarpetas (estructura plana o <cohorte>/)."""
    hits = sorted(Path(folder).rglob(name))
    if not hits: raise FileNotFoundError(f"{name} no esta en {folder}")
    return hits[0]

import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
fm, fe, out = sys.argv[1:4]
THR = {"mob": json.load(open(_f(fm, f"resumen_zeroshot_DDI.json")))["frozen_threshold"],
       "eff": json.load(open(_f(fe, f"resumen_zeroshot_DDI.json")))["frozen_threshold"]}
rows = []
for n in ["DDI", "DDI2", "Fitzpatrick", "PADUFES"]:
    m = pd.read_csv(_f(fm, f"predicciones_zeroshot_{n}.csv")); e = pd.read_csv(_f(fe, f"predicciones_zeroshot_{n}.csv"))
    for x in (m, e): x["name"] = x.filepath.map(lambda s: s.replace("\\", "/").split("/")[-1])
    d = m.merge(e[["name", "y_score"]], on="name", suffixes=("_m", "_e")); assert len(d) == len(m) == len(e)
    if n == "PADUFES":
        d["unit"] = d.name.str.extract(r"PAT_(\d+)_")[0]
    else:
        d["unit"] = d.name
    g = d.groupby("unit").indices; ks = list(g); rng = np.random.default_rng(2026)
    res = {k: [] for k in ["auc_m", "auc_e", "diff", "se_e", "sp_e"]}
    for _ in range(1000):
        b = d.iloc[np.concatenate([g[k] for k in rng.choice(ks, len(ks))])]
        if b.y_true.nunique() < 2: continue
        am, ae = roc_auc_score(b.y_true, b.y_score_m), roc_auc_score(b.y_true, b.y_score_e)
        res["auc_m"].append(am); res["auc_e"].append(ae); res["diff"].append(ae - am)
        p = b.y_score_e >= THR["eff"]
        res["se_e"].append((p & (b.y_true == 1)).sum() / (b.y_true == 1).sum())
        res["sp_e"].append((~p & (b.y_true == 0)).sum() / (b.y_true == 0).sum())
    ci = {k: np.percentile(v, [2.5, 97.5]).round(3).tolist() for k, v in res.items()}
    am, ae = roc_auc_score(d.y_true, d.y_score_m), roc_auc_score(d.y_true, d.y_score_e)
    rows.append({"cohort": n, "unit": "patient" if n == "PADUFES" else "image", "auc_mob": round(am, 3),
                 "auc_eff": round(ae, 3), "auc_eff_ci": ci["auc_e"], "sens_eff_ci": ci["se_e"], "spec_eff_ci": ci["sp_e"],
                 "diff_eff_minus_mob": round(ae - am, 3), "diff_ci": ci["diff"]})
r = pd.DataFrame(rows); print(r.to_string()); r.to_csv(out, index=False)
