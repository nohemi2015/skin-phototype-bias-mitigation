"""Derm7pt: AUC zero-shot de A, B y C40 (media de 3 semillas) en las imagenes
dermatoscopicas y clinicas de las mismas lesiones; bootstrap pareado por caso
(B = 2000); Holm sobre las 4 comparaciones A - C40 (2 backbones x 2 modalidades).
Uso: python analisis_derm7pt.py predicciones_derm7pt.csv verificacion_modelos_finales.csv salida/"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
pred, ver, out = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
thr = dict(zip(ver.modelo, ver.umbral))
B = 2000; rng = np.random.default_rng(2026)
cases = np.sort(pred.case_num.unique()); n = len(cases)
idx = rng.integers(0, n, (B, n))
def cfg_of(m):
    parts = m.split("_"); return parts[1], parts[2]
rows, diffs = [], []
S = {}
for mod in ["dermoscopic", "clinical"]:
    sub = pred[pred.modality == mod]
    w = sub.pivot(index="case_num", columns="model", values="y_score").loc[cases]
    y = sub.drop_duplicates("case_num").set_index("case_num").y_true.loc[cases].values
    for arch in ["mobilenetv2", "efficientnetb0"]:
        for cfg in ["A", "B", "C40", "D"]:
            ms = [m for m in w.columns if cfg_of(m) == (cfg, arch)]
            assert len(ms) == 3, (cfg, arch, ms)
            aucs = [roc_auc_score(y, w[m].values) for m in ms]
            boot = np.array([[roc_auc_score(y[i], w[m].values[i]) for m in ms] for i in idx]).mean(1)
            S[(mod, arch, cfg)] = (np.mean(aucs), boot)
            sens = [((w[m].values >= thr[m]) & (y == 1)).sum() / (y == 1).sum() for m in ms]
            spec = [((w[m].values < thr[m]) & (y == 0)).sum() / (y == 0).sum() for m in ms]
            rows.append(dict(modality=mod, arch=arch, cfg=cfg, auc=np.mean(aucs), lo=np.percentile(boot, 2.5),
                             hi=np.percentile(boot, 97.5), seed_min=min(aucs), seed_max=max(aucs),
                             sens_frozen=np.mean(sens), spec_frozen=np.mean(spec)))
def pval(d): return max(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / B)
for arch in ["mobilenetv2", "efficientnetb0"]:
    for cfg in ["A", "B", "C40", "D"]:
        a, b = S[("dermoscopic", arch, cfg)], S[("clinical", arch, cfg)]; d = a[1] - b[1]
        diffs.append(dict(comp=f"{cfg}: dermoscopic - clinical", arch=arch, diff=a[0] - b[0],
                          lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p=pval(d)))
    for mod in ["dermoscopic", "clinical"]:
        for x, z in [("A", "C40"), ("A", "B"), ("B", "C40"), ("A", "D")]:
            a, b = S[(mod, arch, x)], S[(mod, arch, z)]; d = a[1] - b[1]
            diffs.append(dict(comp=f"{mod}: {x} - {z}", arch=arch, diff=a[0] - b[0],
                              lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p=pval(d)))
r = pd.DataFrame(rows); dd = pd.DataFrame(diffs)
def holm(p):
    o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj
dd["p_holm"] = np.nan
for fam in ["A - C40", "A - D", "dermoscopic - clinical"]:
    mk = dd.comp.str.contains(fam)
    dd.loc[mk, "p_holm"] = holm(dd.loc[mk, "p"].values)
r.to_csv(out / "auc_derm7pt.csv", index=False); dd.to_csv(out / "diff_derm7pt.csv", index=False)
pd.set_option("display.width", 200)
print(r.round(3).to_string(index=False)); print(dd.round(4).to_string(index=False))
