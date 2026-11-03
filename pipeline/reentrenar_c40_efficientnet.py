"""
reentrenar_c40_efficientnet.py  (2-nov-2026)
============================================
Regenera los 3 modelos finales EfficientNet-B0 C40 (semillas 2026, 2027, 2028) y los
evalua SIN recargar el archivo .keras.

Motivo (auditar_c40_efficientnet.py): al recargar FINAL_C40_efficientnetb0*.keras con
keras.models.load_model, el AUC en la validacion del desarrollo baja de 0.785 (en memoria,
historial de entrenamiento) a 0.742-0.746; en MobileNetV2 la recarga es exacta. Los modelos
de la validacion cruzada se evaluaron en memoria y no tienen el problema.

Que hace, para cada semilla (mismo split, mismo procedimiento y mismos hiperparametros que
train_final de entrenar_leakage_safe_v4.py):
  1. Entrena el modelo fuente A (HAM10000) y luego C40 (ultimas 40 capas) sobre Brown Skin.
     El mejor checkpoint (val_auc) se guarda como PESOS (.weights.h5) y se carga en el
     MISMO objeto en memoria, sin reconstruir el modelo.
  2. Comprueba la carga: (a) AUC en memoria en la validacion Brown Skin del desarrollo,
     comparado con el mejor val_auc del historial; (b) un modelo reconstruido con
     build_model + unfreeze_last_layers + load_weights debe dar las mismas predicciones;
     (c) se registra que da keras.models.load_model del .keras, para documentar el fallo.
  3. Con el modelo verificado: umbral congelado (Youden en la validacion Brown Skin),
     control Brown Skin del hold-out, las cuatro cohortes externas y Derm7pt (dermatoscopica
     y clinica), con los mismos formatos de archivo que los scripts anteriores.

Uso (desde la carpeta del proyecto, la de entrenar_leakage_safe_v4.py; tarda varias horas):
    python reentrenar_c40_efficientnet.py --derm7pt_dir "C:\\1. HAM10000\\derm7pt"
Opcional: --semillas 2026 2027 2028
Salida: resultados\\leakage_safe\\c40_eff_v2\\   (mandar esa carpeta comprimida)
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras

import entrenar_leakage_safe_v4 as E
from evaluar_zeroshot_final_v2 import bootstrap, sens_spec
from evaluar_derm7pt import find_meta, load_cases

ARCH = "efficientnetb0"
COHORTES = {"DDI": E.DATA_DIR / "DDI_clasificado", "DDI2": E.DATA_DIR / "DDI2_clasificado",
            "Fitzpatrick": E.DATA_DIR / "Fitzpatrick_clasificado", "PADUFES": E.DATA_DIR / "PADUFES_clasificado"}
OUT = E.PROJECT_DIR / "resultados" / "leakage_safe" / "c40_eff_v2"
MOD = E.PROJECT_DIR / "modelos" / "leakage_safe" / "c40_eff_v2"


class BestInRAM(keras.callbacks.Callback):
    """Guarda en memoria (arrays numpy) los pesos de la mejor epoca segun val_auc."""
    def __init__(self):
        super().__init__(); self.best = -np.inf; self.weights = None; self.epoch = None
    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get("val_auc")
        if v is not None and v > self.best:
            self.best, self.epoch = v, epoch
            self.weights = [w.copy() for w in self.model.get_weights()]


def train_diag(model, train_df, val_df, wpath, hpath, epochs):
    """Entrena con los mismos callbacks que entrenar_leakage_safe_v4 (mas BestInRAM) y NO recarga nada."""
    ram = BestInRAM()
    es = keras.callbacks.EarlyStopping(monitor="val_auc", patience=6, mode="max",
                                       restore_best_weights=True, verbose=1)
    cbs = [es, keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                                 min_lr=1e-7, verbose=1),
           keras.callbacks.ModelCheckpoint(str(wpath), monitor="val_auc", mode="max",
                                           save_best_only=True, save_weights_only=True, verbose=1),
           keras.callbacks.CSVLogger(str(hpath), append=False), ram]
    tr = E.make_dataset(train_df, augment=True, shuffle=True)
    va = E.make_dataset(val_df, augment=False, shuffle=False)
    model.fit(tr, validation_data=va, epochs=epochs, callbacks=cbs, verbose=2)
    return model, ram, int(es.stopped_epoch)


def predict(model, df):
    y, s = E.predict_scores(model, E.make_dataset(df[["filepath", "label"]], augment=False, shuffle=False))
    return np.asarray(y), np.asarray(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derm7pt_dir", required=True)
    ap.add_argument("--semillas", type=int, nargs="+", default=[2026, 2027, 2028])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); MOD.mkdir(parents=True, exist_ok=True)
    print("Version de Keras:", keras.__version__)

    # ---- mismo split que train_final ----
    E.set_seeds(E.BASE_SEED)
    ham_df = E.collect_image_paths(E.HAM_DIR, "HAM_claras")
    bs_df = E.collect_image_paths(E.BS_DIR, "BS_oscuras")
    prov = E.build_provenance_map(E.HAM_METADATA_PATH)
    ham_df = E.add_lesion_id(ham_df, prov, "HAM"); bs_df = E.add_lesion_id(bs_df, prov, "BS")
    h = ham_df.copy(); h["source"] = "ham"; b = bs_df.copy(); b["source"] = "bs"
    comb = pd.concat([h, b], ignore_index=True)
    dev_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.15, random_state=E.BASE_SEED).split(comb, groups=comb["lesion_id"]))
    dev = comb.iloc[dev_idx].reset_index(drop=True); test = comb.iloc[test_idx].reset_index(drop=True)
    tr_idx, va_idx = next(GroupShuffleSplit(1, test_size=0.15, random_state=E.BASE_SEED + 1).split(dev, groups=dev["lesion_id"]))
    trn = dev.iloc[tr_idx].reset_index(drop=True); val = dev.iloc[va_idx].reset_index(drop=True)
    ham_tr, ham_va = trn[trn.source == "ham"], val[val.source == "ham"]
    bs_tr, bs_va = trn[trn.source == "bs"], val[val.source == "bs"]
    bs_test = test[test.source == "bs"].reset_index(drop=True)
    print(f"train {len(trn)} | val {len(val)} | hold-out BS {len(bs_test)}")
    ext = {k: E.collect_image_paths(v, k) for k, v in COHORTES.items()}
    cases = load_cases(a.derm7pt_dir, find_meta(a.derm7pt_dir))

    log = []
    for seed in a.semillas:
        sfx = "" if seed == E.BASE_SEED else f"_s{seed}"
        tag = f"C40_{ARCH}{sfx}"
        print(f"\n===== {tag} =====")
        E.set_seeds(seed)
        ma = E.build_model(ARCH, train_base=False)
        ma, _, _ = train_diag(ma, ham_tr, ham_va, MOD / f"FINAL_A_base_{ARCH}{sfx}.weights.h5",
                              OUT / f"hist_FINAL_A_base_{ARCH}{sfx}.csv", E.EPOCHS_BASE)
        m = keras.models.clone_model(ma); m.set_weights(ma.get_weights())
        m = E.unfreeze_last_layers(m, n_layers=40)
        wpath = MOD / f"FINAL_{tag}.weights.h5"
        m, ram, stopped = train_diag(m, bs_tr, bs_va, wpath, OUT / f"hist_FINAL_{tag}.csv", E.EPOCHS_FT)
        hist = pd.read_csv(OUT / f"hist_FINAL_{tag}.csv")

        # ---- diagnostico: que pesos reproducen el val_auc del historial ----
        yv, s_end = predict(m, bs_va)                      # pesos al terminar fit (sin recargar nada)
        end_w = [w.copy() for w in m.get_weights()]
        m.set_weights(ram.weights); _, s_ram = predict(m, bs_va)      # mejor epoca guardada en RAM
        m.load_weights(str(wpath)); _, s_ck = predict(m, bs_va)       # checkpoint del archivo
        chk = dict(modelo=f"FINAL_{tag}", early_stop_epoch=stopped,
                   val_auc_hist_ultima=float(hist.val_auc.iloc[-1]), val_auc_hist_mejor=float(hist.val_auc.max()),
                   epoca_mejor=int(ram.epoch),
                   auc_fin_fit_en_memoria=roc_auc_score(yv, s_end),
                   auc_mejor_epoca_ram=roc_auc_score(yv, s_ram),
                   auc_checkpoint_archivo=roc_auc_score(yv, s_ck),
                   dif_ram_vs_checkpoint=float(np.abs(s_ram - s_ck).max()))
        m.set_weights(end_w)
        ev = m.evaluate(E.make_dataset(bs_va[["filepath", "label"]], augment=False, shuffle=False),
                        verbose=0, return_dict=True)
        chk["auc_keras_evaluate_fin_fit"] = float(ev["auc"])    # misma metrica que val_auc del historial
        print(chk); log.append(chk); pd.DataFrame(log).to_csv(OUT / "comprobacion_carga.csv", index=False)

        # modelo que se evalua: la mejor epoca en RAM si reproduce el historial; si no, el de fin de fit
        if abs(chk["auc_mejor_epoca_ram"] - chk["val_auc_hist_mejor"]) < 0.01:
            m.set_weights(ram.weights); usado = "mejor_epoca_ram"
        elif abs(chk["auc_fin_fit_en_memoria"] - chk["val_auc_hist_ultima"]) < 0.01:
            m.set_weights(end_w); usado = "fin_fit_en_memoria"
        else:
            print("\n*** Ningun conjunto de pesos reproduce el val_auc del historial. Se detiene aqui ***")
            print(f"Manda la carpeta {OUT} (comprobacion_carga.csv)")
            return
        log[-1]["pesos_evaluados"] = usado
        np.savez(MOD / f"FINAL_{tag}_pesos.npz", *m.get_weights())
        r = E.build_model(ARCH, train_base=False); r = E.unfreeze_last_layers(r, n_layers=40)
        z = np.load(MOD / f"FINAL_{tag}_pesos.npz"); r.set_weights([z[k] for k in z.files])
        _, s_r = predict(r, bs_va); yv2, s_m = predict(m, bs_va); del r
        log[-1]["dif_reconstruido_npz"] = float(np.abs(s_r - s_m).max())
        pd.DataFrame(log).to_csv(OUT / "comprobacion_carga.csv", index=False)
        sv = s_m

        # ---- evaluacion con el modelo en memoria ----
        thr = E.youden_threshold(yv, sv)
        yt, st = predict(m, bs_test)
        pd.DataFrame({"y_true": yt, "y_score": st, "filepath": bs_test.filepath.values,
                      "lesion_id": bs_test.lesion_id.values, "source": "bs", "model": "FINAL_C40",
                      "partition": "test_holdout"}).to_csv(OUT / f"pred_FINAL_{tag}_test_BS.csv", index=False)
        ctrl_auc = roc_auc_score(yt, st)
        info = {"model_file": str(MOD / f"FINAL_{tag}_pesos.npz"), "pesos_evaluados": usado, "arch": ARCH, "config": "C40", "train_seed": seed,
                "frozen_threshold_from_bs_val": thr, "control_bs_auc": ctrl_auc,
                "n_test_bs": int(len(bs_test)), "nota": "evaluado en memoria; pesos en .weights.h5"}
        json.dump(info, open(OUT / f"FINAL_{tag}.json", "w"), indent=2)
        print(f"control Brown Skin AUC {ctrl_auc:.4f} | umbral {thr:.4f}")
        for name, df in ext.items():
            y, s = predict(m, df)
            od = OUT / "zeroshot" / f"FINAL_{tag}" / name; od.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"y_true": y, "y_score": s, "filepath": df.filepath.values}).to_csv(
                od / f"predicciones_zeroshot_{name}.csv", index=False)
            sf, pf = sens_spec(y, s, thr); ty = E.youden_threshold(y, s); sy, py = sens_spec(y, s, ty)
            ci = bootstrap(y, s, thr); auc = float(roc_auc_score(y, s))
            json.dump({"dataset": name, "n": int(len(y)), "n_benign": int((y == 0).sum()), "n_malignant": int((y == 1).sum()),
                       "frozen_model_file": str(MOD / f"FINAL_{tag}_pesos.npz"), "auc": auc, "auc_ci95": ci["auc"], "frozen_threshold": thr,
                       "sensitivity_frozen": sf, "sensitivity_frozen_ci95": ci["sens_frozen"],
                       "specificity_frozen": pf, "specificity_frozen_ci95": ci["spec_frozen"],
                       "youden_threshold_on_external": ty, "sensitivity_youden": sy, "sensitivity_youden_ci95": ci["sens_youden"],
                       "specificity_youden": py, "specificity_youden_ci95": ci["spec_youden"],
                       "control_bs_auc": ctrl_auc, "domain_gap_vs_control": ctrl_auc - auc, "n_bootstrap": 1000},
                      open(od / f"resumen_zeroshot_{name}.json", "w"), indent=2)
            print(f"  {name}: AUC {auc:.4f}")
        parts = []
        for mod, col in (("dermoscopic", "derm_path"), ("clinical", "clinic_path")):
            df = pd.DataFrame({"filepath": cases[col].values, "label": cases.label.values})
            y, s = predict(m, df)
            parts.append(pd.DataFrame({"model": f"FINAL_{tag}", "case_num": cases.case_num.values,
                                       "modality": mod, "y_true": y, "y_score": s}))
            print(f"  Derm7pt {mod}: AUC {roc_auc_score(y, s):.4f}")
        pf_ = OUT / "predicciones_derm7pt_c40_eff.csv"
        pd.concat(parts).to_csv(pf_, mode="a", header=not pf_.exists(), index=False)
        pd.DataFrame(log).to_csv(OUT / "comprobacion_carga.csv", index=False)
        del m, ma; keras.backend.clear_session()

    print(pd.DataFrame(log).round(4).to_string(index=False))
    print(f"\nListo. Comprime y manda la carpeta: {OUT}")


if __name__ == "__main__":
    main()
