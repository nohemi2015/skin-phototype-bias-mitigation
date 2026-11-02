"""Fase 2/3/5: AUC zero-shot de los modelos finales A, B, C40, D y E (3 semillas cada uno) por backbone y cohorte,
diferencias pareadas de las medias de semillas por bootstrap (PAD-UFES-20 remuestreado por paciente),
ajuste de Holm sobre las 8 comparaciones backbone-cohorte, y el control Brown Skin (bootstrap por lesión).

Uso: python analisis_fase2.py <carpeta_zeroshot> <carpeta_control_brownskin> <salida>
  <carpeta_zeroshot>/FINAL_<cfg>_<arch>[_s<seed>]/<cohorte>/predicciones_zeroshot_<cohorte>.csv
  <carpeta_control_brownskin>/pred_FINAL_<cfg>_<arch>[_s<seed>]_test_BS.csv
Salidas: auc_fase2.csv, diff_fase2.csv, A_vs_C40_holm.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT, CTRL, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
B = 2000
COH = ["DDI", "DDI2", "Fitzpatrick", "PADUFES"]
CFGS = ["A", "B", "C40", "D", "E"]
SEEDS = [2026, 2027, 2028]
COMPS = [("A_mean", "C40_mean"), ("B_mean", "C40_mean"), ("A_mean", "B_mean"),
         ("A_mean", "D_mean"), ("A_mean", "E_mean")]


def name(cfg, arch, seed):
    return f"FINAL_{cfg}_{arch}" + ("" if seed == 2026 else f"_s{seed}")


def boot(y, S, unit, seed=2026):
    units = np.unique(unit)
    pos = {u: np.where(unit == u)[0] for u in units}
    rng = np.random.default_rng(seed)
    bs = {k: [] for k in S}
    for _ in range(B):
        ii = np.concatenate([pos[u] for u in rng.choice(units, len(units))])
        if len(set(y[ii])) < 2:
            continue
        for k in S:
            bs[k].append(roc_auc_score(y[ii], S[k][ii]))
    return {k: np.array(v) for k, v in bs.items()}


def summarize(arch, cohort, y, S, unit):
    bs = boot(y, S, unit)
    auc = {k: roc_auc_score(y, S[k]) for k in S}
    rows, diffs = [], []
    for cfg in CFGS:
        ks = [f"{cfg}_{s}" for s in SEEDS]
        auc[f"{cfg}_mean"] = np.mean([auc[k] for k in ks])
        bs[f"{cfg}_mean"] = np.mean([bs[k] for k in ks], axis=0)
        seed_vals = [auc[k] for k in ks]
        for k in ks + [f"{cfg}_mean"]:
            rows.append(dict(arch=arch, cohort=cohort, model=k, auc=auc[k],
                             lo=np.percentile(bs[k], 2.5), hi=np.percentile(bs[k], 97.5),
                             seed_min=min(seed_vals), seed_max=max(seed_vals)))
    for a, b in COMPS:
        d = bs[a] - bs[b]
        p = min(1, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        diffs.append(dict(arch=arch, cohort=cohort, comp=f"{a} - {b}", diff=auc[a] - auc[b],
                          lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p=max(p, 1 / B)))
    return rows, diffs


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj


rows, diffs = [], []
for arch in ["mobilenetv2", "efficientnetb0"]:
    for c in COH:
        dfs = {f"{cfg}_{s}": pd.read_csv(ROOT / name(cfg, arch, s) / c / f"predicciones_zeroshot_{c}.csv")
               .assign(key=lambda d: d.filepath.str.replace("\\", "/").str.split("/").str[-2:].str.join("/"))
               .set_index("key") for cfg in CFGS for s in SEEDS}
        base = dfs["A_2026"]; idx = base.index
        for k, d in dfs.items():
            assert set(d.index) == set(idx) and (d.loc[idx, "y_true"] == base.y_true).all(), (arch, c, k)
        y = base.y_true.values
        S = {k: d.loc[idx, "y_score"].values for k, d in dfs.items()}
        unit = (pd.Series(idx).str.extract(r"(PAT_\d+)")[0].values if c == "PADUFES" else np.arange(len(idx)))
        r_, d_ = summarize(arch, c, y, S, unit); rows += r_; diffs += d_
    # Control Brown Skin: hold-out de 478 lesiones, bootstrap por lesión
    dfs = {f"{cfg}_{s}": pd.read_csv(CTRL / f"pred_{name(cfg, arch, s)}_test_BS.csv")
           .assign(key=lambda d: d.filepath.str.replace("\\", "/").str.split("/").str[-1]).set_index("key")
           for cfg in CFGS for s in SEEDS}
    base = dfs["A_2026"]; idx = base.index
    for k, d in dfs.items():
        assert set(d.index) == set(idx), k
    y = base.y_true.values
    S = {k: d.loc[idx, "y_score"].values for k, d in dfs.items()}
    r_, d_ = summarize(arch, "BrownSkin_control", y, S, base.lesion_id.values); rows += r_; diffs += d_

r = pd.DataFrame(rows); dd = pd.DataFrame(diffs)
dd["p_holm"] = np.nan
for comp in dd.comp.unique():
    m = (dd.comp == comp) & (dd.cohort != "BrownSkin_control")
    dd.loc[m, "p_holm"] = holm(dd.loc[m, "p"].values)
r.to_csv(OUT / "auc_fase2.csv", index=False)
dd.to_csv(OUT / "diff_fase2.csv", index=False)
dd[dd.comp == "A_mean - C40_mean"].to_csv(OUT / "A_vs_C40_holm.csv", index=False)
pd.set_option("display.width", 250)
print(r[r.model.str.endswith("mean")].pivot_table(index=["arch", "model"], columns="cohort", values="auc").round(3))
print(r[r.model.str.endswith("mean")].pivot_table(index=["arch", "model"], columns="cohort", values=["seed_min", "seed_max"]).round(3))
print(dd.round(4).to_string(index=False))
