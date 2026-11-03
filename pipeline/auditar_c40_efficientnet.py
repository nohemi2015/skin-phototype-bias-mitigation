"""
auditar_c40_efficientnet.py  (2-nov-2026)
=========================================
Auditoria de la discrepancia de EfficientNet-B0 C40 (no entrena nada).

Hallazgo previo (sin pesos, solo con predicciones archivadas): los modelos C40 de
EfficientNet-B0 de la validacion cruzada, evaluados sobre las MISMAS 1,763 imagenes
del hold-out (cada imagen por el modelo del fold en que fue validacion), dan AUC 0.788;
los modelos finales dan 0.703-0.709. El hold-out no es mas dificil: la diferencia esta
en los modelos finales.

Diferencia de codigo entre ambos casos: en la validacion cruzada el modelo se evalua
EN MEMORIA tras entrenar; el modelo final se guarda, se RECARGA del archivo .keras y
despues se evalua. Este script prueba si recargar el archivo cambia las predicciones.

Prueba 1. Modelos C40 de la validacion cruzada (repeticion 1, folds 1-5), ambos
          backbones: se recargan del .keras, se predice su fold de validacion y se
          compara con las predicciones archivadas (que se hicieron en memoria).
Prueba 2. Modelos finales C40 (3 semillas, ambos backbones): se recargan y se evalua
          el subconjunto de validacion Brown Skin del split de desarrollo, el mismo
          que monitoreo el entrenamiento; se compara con el mejor val_auc del historial.

Si en EfficientNet-B0 el AUC recargado es claramente menor que el archivado (prueba 1)
o que el del historial (prueba 2), la causa es la recarga del modelo.

Uso (desde la carpeta del proyecto, la de entrenar_leakage_safe_v4.py):
    python auditar_c40_efficientnet.py
Salida: resultados/leakage_safe/auditoria_c40/auditoria_c40.csv
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras

import entrenar_leakage_safe_v4 as E

ROOTS = [E.PROJECT_DIR / "rerun_cv", E.PROJECT_DIR]   # primero la re-ejecucion de MobileNetV2
OUT = E.PROJECT_DIR / "resultados" / "leakage_safe" / "auditoria_c40"
OUT.mkdir(parents=True, exist_ok=True)


def find(pattern):
    for r in ROOTS:
        hits = glob.glob(str(r / pattern))
        if hits:
            return Path(sorted(hits)[-1])
    return None


def auc_reloaded(model_path, df):
    m = keras.models.load_model(model_path)
    y, s = E.predict_scores(m, E.make_dataset(df, augment=False, shuffle=False))
    keras.backend.clear_session()
    return np.asarray(y), np.asarray(s)


rows = []
print("Version de TensorFlow/Keras:", keras.__version__)

# ---------- Prueba 1: modelos de la validacion cruzada ----------
ledger_path = find("resultados/leakage_safe/split_ledger.csv")
L = pd.read_csv(ledger_path)
for arch, tag in (("mobilenetv2", ""), ("efficientnetb0", "_efficientnetb0")):
    for f in range(1, 6):
        mp = find(f"modelos/leakage_safe/C40_{arch}_rep1_fold{f}.keras")
        pp = find(f"resultados/leakage_safe/predicciones/pred_C40{tag}_rep1_fold{f}.csv")
        if mp is None or pp is None:
            print(f"[falta] C40 {arch} rep1 fold{f}: modelo={mp} pred={pp}")
            continue
        v = L[(L.source == "bs") & (L.partition == "val") & (L.repeat == 1) & (L.fold == f)]
        df = pd.DataFrame({"filepath": v.filepath.values, "label": v.label.values})
        p = pd.read_csv(pp)
        assert len(p) == len(df) and (p.y_true.values == df.label.values).all()
        y, s = auc_reloaded(mp, df)
        r = dict(prueba="1_cv_fold", arch=arch, modelo=mp.name,
                 auc_archivado_en_memoria=roc_auc_score(p.y_true, p.y_score),
                 auc_recargado=roc_auc_score(y, s),
                 max_dif_score=float(np.abs(s - p.y_score.values).max()))
        rows.append(r); print(r)

# ---------- Prueba 2: modelos finales, subconjunto de validacion del desarrollo ----------
ham_df = E.collect_image_paths(E.HAM_DIR, "HAM_claras")
bs_df = E.collect_image_paths(E.BS_DIR, "BS_oscuras")
prov = E.build_provenance_map(E.HAM_METADATA_PATH)
ham_df = E.add_lesion_id(ham_df, prov, "HAM"); bs_df = E.add_lesion_id(bs_df, prov, "BS")
h = ham_df.copy(); h["source"] = "ham"; b = bs_df.copy(); b["source"] = "bs"
comb = pd.concat([h, b], ignore_index=True)
dev_idx, _ = next(GroupShuffleSplit(1, test_size=0.15, random_state=E.BASE_SEED).split(comb, groups=comb["lesion_id"]))
dev = comb.iloc[dev_idx].reset_index(drop=True)
_, va_idx = next(GroupShuffleSplit(1, test_size=0.15, random_state=E.BASE_SEED + 1).split(dev, groups=dev["lesion_id"]))
bs_va = dev.iloc[va_idx]; bs_va = bs_va[bs_va.source == "bs"].reset_index(drop=True)
print(f"Validacion Brown Skin del desarrollo: {len(bs_va)} imagenes")
for arch in ("mobilenetv2", "efficientnetb0"):
    for sfx in ("", "_s2027", "_s2028"):
        mp = find(f"modelos/leakage_safe/FINAL_C40_{arch}{sfx}.keras")
        hp = find(f"resultados/leakage_safe/hist_FINAL_C40_{arch}{sfx}.csv")
        if mp is None:
            print(f"[falta] FINAL_C40_{arch}{sfx}.keras"); continue
        y, s = auc_reloaded(mp, bs_va[["filepath", "label"]])
        best = float(pd.read_csv(hp).val_auc.max()) if hp is not None else np.nan
        r = dict(prueba="2_final_val_desarrollo", arch=arch, modelo=mp.name,
                 auc_archivado_en_memoria=best, auc_recargado=roc_auc_score(y, s), max_dif_score=np.nan)
        rows.append(r); print(r)

out = pd.DataFrame(rows)
out.to_csv(OUT / "auditoria_c40.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))
print(f"\nListo: {OUT / 'auditoria_c40.csv'}")
