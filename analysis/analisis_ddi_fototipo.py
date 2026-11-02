"""AUC zero-shot por grupo de fototipo en DDI (I-II, III-IV, V-VI).
Dos variantes: 'all' (todas las malignidades de DDI son positivas) y 'MEL/BCC/SCC'
(positivos restringidos a melanoma, carcinoma basocelular y espinocelular; se conservan
todas las benignas). IC 95 % por bootstrap de imagenes dentro de cada grupo (B = 2000).
Uso: python analisis_ddi_fototipo.py <ddi_metadata.csv> <salida.csv> NOMBRE=predicciones.csv:umbral [...]"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
B, SEED = 2000, 2026
GROUPS = {12: "I-II", 34: "III-IV", 56: "V-VI"}
HAM_TYPES = ("melanoma", "basal-cell", "squamous-cell")
meta = pd.read_csv(sys.argv[1]); out = sys.argv[2]
meta["file"] = meta["DDI_file"].str.strip()
meta["ham_type"] = meta["disease"].str.contains("|".join(HAM_TYPES))
rows, diffs = [], []
for arg in sys.argv[3:]:
    name, rest = arg.split("=", 1); path, thr = rest.rsplit(":", 1); thr = float(thr)
    p = pd.read_csv(path)
    p["file"] = p["filepath"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    d0 = p.merge(meta, on="file", how="left", validate="1:1")
    assert d0["skin_tone"].notna().all(), "imagenes sin metadata"
    assert (d0["y_true"] == d0["malignant"].astype(int)).all(), "etiquetas no coinciden"
    for variant in ["all", "MEL/BCC/SCC"]:
        d = d0 if variant == "all" else d0[(d0.y_true == 0) | d0.ham_type]
        rng = np.random.default_rng(SEED)
        y, s, tone = d.y_true.values, d.y_score.values, d.skin_tone.values
        idx = {g: np.where(tone == g)[0] for g in GROUPS}
        boots = {g: [] for g in GROUPS}
        for _ in range(B):
            for g, ii in idx.items():
                b = rng.choice(ii, len(ii))
                boots[g].append(roc_auc_score(y[b], s[b]) if len(set(y[b])) == 2 else np.nan)
        for g, lab in GROUPS.items():
            ii = idx[g]; yy, ss = y[ii], s[ii]; pr = ss >= thr
            rows.append({"model": name, "positives": variant, "group": lab, "n": len(ii), "n_mal": int(yy.sum()),
                         "auc": roc_auc_score(yy, ss),
                         "auc_lo": np.nanpercentile(boots[g], 2.5), "auc_hi": np.nanpercentile(boots[g], 97.5),
                         "sens_frozen": pr[yy == 1].mean(), "spec_frozen": 1 - pr[yy == 0].mean()})
        dd = np.array(boots[56]) - np.array(boots[12])
        diffs.append({"model": name, "positives": variant,
                      "auc_VVI_minus_III": roc_auc_score(y[idx[56]], s[idx[56]]) - roc_auc_score(y[idx[12]], s[idx[12]]),
                      "lo": np.nanpercentile(dd, 2.5), "hi": np.nanpercentile(dd, 97.5)})
r = pd.DataFrame(rows); r.to_csv(out, index=False)
pd.DataFrame(diffs).to_csv(out.replace(".csv", "_diferencias.csv"), index=False)
pd.set_option("display.width", 200)
print(r.round(3).to_string(index=False)); print(pd.DataFrame(diffs).round(3).to_string(index=False))
