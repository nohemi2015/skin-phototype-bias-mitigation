"""
p1b_comparar_c40ham.py - compara A, C40 (Brown Skin) y C40-HAM con el mismo bootstrap pareado
del articulo (B = 2000; lesiones para Derm7pt, pacientes para PAD-UFES-20, imagenes en el resto;
media de las 3 semillas; Holm por familia). No necesita GPU: solo lee archivos de prediccion.

Uso:  python p1b_comparar_c40ham.py --repo_results results   (carpeta results/ del repositorio)
"""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

SFX = ["", "_s2027", "_s2028"]
OUT = Path("resultados") / "leakage_safe" / "c40_ham"   # salida de p1_c40_solo_ham.py


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def preds(R, cfg, arch, cohort):
    out = []
    for s in SFX:
        tag = f"FINAL_{cfg}_{arch}{s}"
        if cfg == "C40HAM":
            base = OUT
            if cohort.startswith("Derm7pt"):
                d = pd.read_csv(OUT / "predicciones_derm7pt_c40ham.csv")
                d = d[(d.model == tag) & (d.modality == cohort.split("_")[1])].sort_values("case_num")
                out.append(d); continue
            d = pd.read_csv(base / "zeroshot" / tag / cohort / f"predicciones_zeroshot_{cohort}.csv")
        else:
            if cohort.startswith("Derm7pt"):
                d = pd.read_csv(R / "derm7pt" / "predicciones_derm7pt.csv")
                d = d[(d.model == tag) & (d.modality == cohort.split("_")[1])].sort_values("case_num")
                out.append(d); continue
            d = pd.read_csv(R / "zeroshot" / tag / cohort / f"predicciones_zeroshot_{cohort}.csv")
        d["name"] = d.filepath.astype(str).str.replace("\\\\", "/", regex=True).str.split("/").str[-1]
        out.append(d.sort_values("name"))
    return out


def unidades(d, cohort):
    if cohort == "PADUFES":
        return d.name.str.extract(r"PAT_(\d+)_")[0].values
    if cohort.startswith("Derm7pt"):
        return d.case_num.values
    return np.arange(len(d))


def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--repo_results", required=True); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--c40ham_dir", default=str(OUT))
    a = ap.parse_args(); OUT = Path(a.c40ham_dir); R = Path(a.repo_results); rows = []
    for arch in ["mobilenetv2", "efficientnetb0"]:
        for cohort in ["DDI", "DDI2", "Fitzpatrick", "PADUFES", "Derm7pt_dermoscopic", "Derm7pt_clinical"]:
            P = {c: preds(R, c, arch, cohort) for c in ["A", "C40", "C40HAM"]}
            y = P["A"][0].y_true.values
            for c in P:
                for d in P[c]:
                    assert (d.y_true.values == y).all(), (c, cohort, "orden distinto")
            u = unidades(P["A"][0], cohort); g = pd.Series(np.arange(len(y))).groupby(u).indices; ks = list(g)
            rng = np.random.default_rng(2026); boot = {k: [] for k in ["A-C40HAM", "C40HAM-C40"]}
            mean = lambda c, idx: np.mean([roc_auc_score(y[idx], d.y_score.values[idx]) for d in P[c]])
            full = np.arange(len(y)); est = {c: mean(c, full) for c in P}
            for _ in range(a.B):
                idx = np.concatenate([g[k] for k in rng.choice(ks, len(ks))])
                m = {c: mean(c, idx) for c in P}
                boot["A-C40HAM"].append(m["A"] - m["C40HAM"]); boot["C40HAM-C40"].append(m["C40HAM"] - m["C40"])
            for k, v in boot.items():
                v = np.asarray(v); p = min(1, 2 * min((v <= 0).mean(), (v >= 0).mean())); p = max(p, 1 / a.B)
                f, s = k.split("-")
                rows.append(dict(arch=arch, cohort=cohort, comparison=k, auc_first=est[f], auc_second=est[s],
                                 delta=est[f] - est[s], lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5), p=p))
    r = pd.DataFrame(rows)
    # familias como en el articulo: 8 comparaciones clinicas y 4 de Derm7pt, por tipo de comparacion
    r["familia"] = r.comparison + np.where(r.cohort.str.startswith("Derm7pt"), "_derm7pt", "_clinicas")
    r["p_holm"] = r.groupby("familia").p.transform(lambda s: holm(s.values))
    r.to_csv(OUT / "comparacion_c40ham_B2000.csv", index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
