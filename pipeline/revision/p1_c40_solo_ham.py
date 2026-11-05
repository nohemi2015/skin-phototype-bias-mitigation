"""
p1_c40_solo_ham.py - CONTROL PENDIENTE PRINCIPAL (revision, Discusion 5.2).

Pregunta: la caida de C40 en imagenes reales (Derm7pt dermatoscopica, Fitzpatrick17k, PAD-UFES-20)
se debe a Brown Skin o al ajuste fino del backbone en si?
Diseno: igual que C40 final, pero las ultimas 40 capas se ajustan sobre HAM10000 (entrenamiento y
validacion de HAM10000 del mismo split final), sin ninguna imagen de Brown Skin.
  - Si C40-HAM tambien cae frente a A -> la perdida se atribuye al ajuste fino del backbone.
  - Si C40-HAM se mantiene como A -> la perdida se atribuye a Brown Skin.
Umbral congelado: Youden en la validacion de Brown Skin, como en los demas modelos finales.
Salida: resultados/leakage_safe/c40_ham/ con los mismos formatos de prediccion que el resto.

Uso:  python p1_c40_solo_ham.py --derm7pt_dir "C:\\1. HAM10000\\derm7pt"  [--arch mobilenetv2 efficientnetb0]
Duracion aproximada: similar a reentrenar_c40_efficientnet.py por modelo (CPU, varias horas).
"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from tensorflow import keras
import entrenar_leakage_safe_v4 as E
from evaluar_derm7pt import find_meta, load_cases
from comun import COHORTES, SEMILLAS, split_final, entrenar, predecir

OUT = E.PROJECT_DIR / "resultados" / "leakage_safe" / "c40_ham"
MOD = E.PROJECT_DIR / "modelos" / "leakage_safe" / "c40_ham"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derm7pt_dir", required=True)
    ap.add_argument("--arch", nargs="+", default=["mobilenetv2", "efficientnetb0"])
    ap.add_argument("--semillas", type=int, nargs="+", default=SEMILLAS)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); MOD.mkdir(parents=True, exist_ok=True)
    S = split_final()
    ext = {k: E.collect_image_paths(v, k) for k, v in COHORTES.items()}
    cases = load_cases(a.derm7pt_dir, find_meta(a.derm7pt_dir))
    resumen = []
    for arch in a.arch:
        for seed in a.semillas:
            sfx = "" if seed == E.BASE_SEED else f"_s{seed}"
            tag = f"C40HAM_{arch}{sfx}"
            print(f"\n===== {tag} =====")
            E.set_seeds(seed)
            ma = entrenar(E.build_model(arch, train_base=False), S["ham_tr"], S["ham_va"], OUT / f"hist_A_{tag}.csv", E.EPOCHS_BASE)
            m = keras.models.clone_model(ma); m.set_weights(ma.get_weights())
            m = E.unfreeze_last_layers(m, n_layers=40)
            m = entrenar(m, S["ham_tr"], S["ham_va"], OUT / f"hist_{tag}.csv", E.EPOCHS_FT)   # <- solo HAM10000
            np.savez(MOD / f"FINAL_{tag}_pesos.npz", *m.get_weights())
            yv, sv = predecir(m, S["bs_va"]); thr = float(E.youden_threshold(yv, sv))
            row = dict(modelo=tag, arch=arch, seed=seed, umbral=thr)
            yt, st = predecir(m, S["bs_test"])
            pd.DataFrame({"y_true": yt, "y_score": st, "filepath": S["bs_test"].filepath.values,
                          "lesion_id": S["bs_test"].lesion_id.values}).to_csv(OUT / f"pred_FINAL_{tag}_test_BS.csv", index=False)
            row["control_BS"] = roc_auc_score(yt, st)
            for name, df in ext.items():
                y, s = predecir(m, df)
                od = OUT / "zeroshot" / f"FINAL_{tag}" / name; od.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"y_true": y, "y_score": s, "filepath": df.filepath.values}).to_csv(
                    od / f"predicciones_zeroshot_{name}.csv", index=False)
                row[name] = roc_auc_score(y, s)
            parts = []
            for mod, col in (("dermoscopic", "derm_path"), ("clinical", "clinic_path")):
                df = pd.DataFrame({"filepath": cases[col].values, "label": cases.label.values})
                y, s = predecir(m, df)
                parts.append(pd.DataFrame({"model": f"FINAL_{tag}", "case_num": cases.case_num.values,
                                           "modality": mod, "y_true": y, "y_score": s}))
                row[f"Derm7pt_{mod}"] = roc_auc_score(y, s)
            pf = OUT / "predicciones_derm7pt_c40ham.csv"
            pd.concat(parts).to_csv(pf, mode="a", header=not pf.exists(), index=False)
            resumen.append(row); print(row)
            pd.DataFrame(resumen).to_csv(OUT / "resumen_c40ham.csv", index=False)
            del m, ma; keras.backend.clear_session()
    print(pd.DataFrame(resumen).round(3).to_string(index=False))
    print(f"\nListo. Despues ejecutar: python p1b_comparar_c40ham.py")


if __name__ == "__main__":
    main()
