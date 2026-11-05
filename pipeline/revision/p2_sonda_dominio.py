"""
p2_sonda_dominio.py - SONDA DE ATAJO REAL vs SINTETICO (revision, Major 3).

Pregunta: los modelos separan HAM10000 (real) de Brown Skin (sintetico) con sus representaciones,
y esa separacion puede actuar como atajo para la etiqueta? En el entrenamiento, la prevalencia de
malignos es ~20 % en HAM10000 y ~50 % en Brown Skin, asi que "parecer sintetico" se asocia con
"maligno" en D y, de forma distinta, entre A y B/C.

Que hace (solo imagenes del hold-out de lesiones, nunca vistas en entrenamiento):
 1. Extrae el vector de global average pooling de: el backbone ImageNet congelado (comun a A, B, D)
    y de cada modelo C40 final (3 semillas x 2 backbones).
 2. Sonda lineal (regresion logistica, GroupKFold de 5 por lesion_id) que predice el dominio
    (HAM = 0, Brown Skin = 1). Reporta AUC de dominio.
 3. Atajo: AUC de la puntuacion de dominio para la ETIQUETA, dentro de HAM, dentro de Brown Skin y
    en las 4 cohortes reales y Derm7pt; y correlacion de Spearman entre la salida del modelo y la
    puntuacion de dominio en cada cohorte real.
Interpretacion: AUC de dominio ~1 es esperable (estilo); lo relevante es si la direccion de dominio
predice la etiqueta en imagenes reales o se correlaciona con la salida del clasificador.

Uso:  python p2_sonda_dominio.py --derm7pt_dir "C:\\1. HAM10000\\derm7pt"
Salida: resultados/leakage_safe/sonda_dominio/
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tensorflow import keras
import entrenar_leakage_safe_v4 as E
from evaluar_derm7pt import find_meta, load_cases
from comun import COHORTES, SEMILLAS, split_final, cargar_final

OUT = E.PROJECT_DIR / "resultados" / "leakage_safe" / "sonda_dominio"


def extractor(m):
    gap = [l for l in m.layers if isinstance(l, keras.layers.GlobalAveragePooling2D)][0]
    return keras.Model(m.inputs, [gap.output, m.outputs[0]])


def feats(fx, df):
    ds = E.make_dataset(df[["filepath", "label"]], augment=False, shuffle=False)
    F, S = [], []
    for x, _ in ds:
        f, s = fx(x, training=False); F.append(f.numpy()); S.append(s.numpy().ravel())
    return np.concatenate(F), np.concatenate(S)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--derm7pt_dir", required=True)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    S = split_final()
    ho = pd.concat([S["ham_test"].assign(dom=0), S["bs_test"].assign(dom=1)], ignore_index=True)
    reales = {k: E.collect_image_paths(v, k) for k, v in COHORTES.items()}
    cases = load_cases(a.derm7pt_dir, find_meta(a.derm7pt_dir))
    reales["Derm7pt_dermoscopic"] = pd.DataFrame({"filepath": cases.derm_path.values, "label": cases.label.values})
    modelos = [("A_imagenet", "A", arch, 2026) for arch in ["mobilenetv2", "efficientnetb0"]]
    modelos += [(f"C40_s{s}", "C40", arch, s) for arch in ["mobilenetv2", "efficientnetb0"] for s in SEMILLAS]
    rows = []
    for name, cfg, arch, seed in modelos:
        m = cargar_final(cfg, arch, seed); fx = extractor(m)
        X, _ = feats(fx, ho); y_dom = ho.dom.values; y_lab = ho.label.values
        oof = np.zeros(len(ho))
        for tr, te in GroupKFold(5).split(X, y_dom, groups=ho.lesion_id.values):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[tr], y_dom[tr])
            oof[te] = clf.decision_function(X[te])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X, y_dom)
        r = dict(modelo=name, arch=arch, auc_dominio=roc_auc_score(y_dom, oof),
                 auc_etiqueta_desde_dominio_HAM=roc_auc_score(y_lab[y_dom == 0], oof[y_dom == 0]),
                 auc_etiqueta_desde_dominio_BS=roc_auc_score(y_lab[y_dom == 1], oof[y_dom == 1]))
        for k, df in reales.items():
            Xr, sr = feats(fx, df); dr = clf.decision_function(Xr); yr = df.label.values
            r[f"{k}_auc_etiqueta_desde_dominio"] = roc_auc_score(yr, dr)
            r[f"{k}_spearman_salida_vs_dominio"] = spearmanr(sr, dr).correlation
        rows.append(r); print(r)
        pd.DataFrame(rows).to_csv(OUT / "sonda_dominio.csv", index=False)
        keras.backend.clear_session()
    print(pd.DataFrame(rows).round(3).T.to_string())


if __name__ == "__main__":
    main()
