"""Fase 2: AUC zero-shot de A, B y C40 (3 semillas) por backbone y cohorte,
con diferencias pareadas por bootstrap (PAD-UFES-20 por paciente)."""
import sys, re
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); B = 2000
COH = ["DDI", "DDI2", "Fitzpatrick", "PADUFES"]
def load(model, c):
    d = pd.read_csv(ROOT / model / c / f"predicciones_zeroshot_{c}.csv")
    return d.set_index("filepath")
rows, diffs = [], []
for arch in ["mobilenetv2", "efficientnetb0"]:
    names = {"A": f"FINAL_A_{arch}", "B": f"FINAL_B_{arch}", "C40_2026": f"FINAL_C40_{arch}",
             "C40_2027": f"FINAL_C40_{arch}_s2027", "C40_2028": f"FINAL_C40_{arch}_s2028"}
    for c in COH:
        dfs = {k: load(v, c) for k, v in names.items()}
        base = dfs["A"]; idx = base.index
        for k in dfs: assert set(dfs[k].index) == set(idx) and (dfs[k].loc[idx, "y_true"] == base.y_true).all()
        y = base.y_true.values; S = {k: dfs[k].loc[idx, "y_score"].values for k in dfs}
        if c == "PADUFES":
            unit = pd.Series(idx).str.extract(r"(PAT_\d+)")[0].values
        else:
            unit = np.arange(len(idx))
        units = np.unique(unit); pos = {u: np.where(unit == u)[0] for u in units}
        rng = np.random.default_rng(2026)
        bs = {k: [] for k in S}
        for _ in range(B):
            pick = rng.choice(units, len(units))
            ii = np.concatenate([pos[u] for u in pick])
            if len(set(y[ii])) < 2: continue
            for k in S: bs[k].append(roc_auc_score(y[ii], S[k][ii]))
        bs = {k: np.array(v) for k, v in bs.items()}
        auc = {k: roc_auc_score(y, S[k]) for k in S}
        c40 = ["C40_2026", "C40_2027", "C40_2028"]
        auc["C40_mean"] = np.mean([auc[k] for k in c40]); bs["C40_mean"] = np.mean([bs[k] for k in c40], axis=0)
        for k in auc:
            rows.append(dict(arch=arch, cohort=c, model=k, auc=auc[k],
                             lo=np.percentile(bs[k], 2.5), hi=np.percentile(bs[k], 97.5)))
        for a_, b_ in [("A", "C40_mean"), ("A", "B"), ("B", "C40_mean"), ("A", "C40_2026")]:
            d = bs[a_] - bs[b_]
            diffs.append(dict(arch=arch, cohort=c, comp=f"{a_} - {b_}", diff=auc[a_] - auc[b_],
                              lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                              p_two=min(1, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
r = pd.DataFrame(rows); dd = pd.DataFrame(diffs)
r.to_csv(OUT / "auc_fase2.csv", index=False); dd.to_csv(OUT / "diff_fase2.csv", index=False)
pd.set_option("display.width", 220)
print(r.pivot_table(index=["arch", "model"], columns="cohort", values="auc").round(3))
print(dd.round(3).to_string(index=False))
