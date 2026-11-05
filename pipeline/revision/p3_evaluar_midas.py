"""
p3_evaluar_midas.py - EVALUACION ZERO-SHOT EN MRA-MIDAS (revision, Major 2).

MRA-MIDAS (Stanford AIMI; Chiou et al., NEJM AI 2025) tiene imagenes dermatoscopicas y clinicas
de lesiones con biopsia y fototipo de Fitzpatrick. Permite medir el efecto de la modalidad y del
fototipo con imagenes reales dermatoscopicas, que es lo que falta en el articulo.

Paso 1 (manual, una vez): construir midas_indice.csv a partir de los metadatos de MIDAS con columnas
    filepath, label (1 = maligno), modality (dermoscopic | clinical_15cm | clinical_30cm),
    fitzpatrick (1-6 o vacio), patient_id, lesion_id
  usando la MISMA definicion de maligno del articulo (melanoma de todos los subtipos, BCC, SCC
  y carcinoma intraepidermico = 1; nevos, queratosis seborreica, otras benignas = 0). Documentar
  la correspondencia de diagnosticos en el material suplementario; excluir diagnosticos sin
  equivalente y reportar cuantos.
Paso 2: python p3_evaluar_midas.py --indice midas_indice.csv
  Evalua los 24 modelos finales congelados (sin reentrenar), guarda predicciones y calcula AUC por
  modalidad y por grupo de fototipo (I-II, III-IV, V-VI) con bootstrap por paciente (B = 2000).
Salida: resultados/leakage_safe/midas/
"""
import argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from tensorflow import keras
import entrenar_leakage_safe_v4 as E
from comun import SEMILLAS, cargar_final, predecir

OUT = E.PROJECT_DIR / "resultados" / "leakage_safe" / "midas"
GRUPOS = {"I-II": [1, 2], "III-IV": [3, 4], "V-VI": [5, 6]}


def boot_auc(df, B=2000, seed=2026):
    g = df.groupby("patient_id").indices; ks = list(g); rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        idx = np.concatenate([g[k] for k in rng.choice(ks, len(ks))])
        d = df.iloc[idx]
        if d.label.nunique() == 2:
            v.append(d.groupby("model").apply(lambda x: roc_auc_score(x.label, x.y_score)).mean())
    return np.percentile(v, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--indice", required=True); ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(a.indice); assert {"filepath", "label", "modality", "fitzpatrick", "patient_id"} <= set(idx.columns)
    pred_f = OUT / "predicciones_midas.csv"
    if not pred_f.exists():
        parts = []
        for arch in ["mobilenetv2", "efficientnetb0"]:
            for cfg in ["A", "B", "C40", "D"]:
                for s in SEMILLAS:
                    m = cargar_final(cfg, arch, s); y, sc = predecir(m, idx)
                    sfx = "" if s == E.BASE_SEED else f"_s{s}"
                    parts.append(idx.assign(model=f"FINAL_{cfg}_{arch}{sfx}", arch=arch, cfg=cfg, y_score=sc))
                    print(cfg, arch, s, "AUC global", round(roc_auc_score(y, sc), 3)); keras.backend.clear_session()
        pd.concat(parts).to_csv(pred_f, index=False)
    P = pd.read_csv(pred_f); rows = []
    for (arch, cfg), d in P.groupby(["arch", "cfg"]):
        for mod, dm in d.groupby("modality"):
            subsets = {"todos": dm}
            subsets.update({g: dm[dm.fitzpatrick.isin(v)] for g, v in GRUPOS.items()})
            for g, ds in subsets.items():
                if ds.label.nunique() < 2: continue
                auc = ds.groupby("model").apply(lambda x: roc_auc_score(x.label, x.y_score)).mean()
                lo, hi = boot_auc(ds.reset_index(drop=True), a.B)
                rows.append(dict(arch=arch, cfg=cfg, modality=mod, fst=g, n=ds.filepath.nunique(),
                                 n_mal=int(ds.drop_duplicates("filepath").label.sum()), auc_media_semillas=auc, lo=lo, hi=hi))
    r = pd.DataFrame(rows); r.to_csv(OUT / "auc_midas.csv", index=False); print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
