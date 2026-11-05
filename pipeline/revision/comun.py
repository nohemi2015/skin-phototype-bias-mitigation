"""
comun.py - utilidades compartidas por los scripts pendientes de la revision (nov-2026).
Copiar esta carpeta dentro de la carpeta del proyecto (la de entrenar_leakage_safe_v4.py)
y ejecutar los scripts desde alli.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras
import entrenar_leakage_safe_v4 as E

COHORTES = {"DDI": E.DATA_DIR / "DDI_clasificado", "DDI2": E.DATA_DIR / "DDI2_clasificado",
            "Fitzpatrick": E.DATA_DIR / "Fitzpatrick_clasificado", "PADUFES": E.DATA_DIR / "PADUFES_clasificado"}
SEMILLAS = [2026, 2027, 2028]


def split_final():
    """Reproduce exactamente el split de train_final (desarrollo 85 % / hold-out 15 % por lesion)."""
    E.set_seeds(E.BASE_SEED)
    ham = E.add_lesion_id(E.collect_image_paths(E.HAM_DIR, "HAM_claras"), E.build_provenance_map(E.HAM_METADATA_PATH), "HAM")
    bs = E.add_lesion_id(E.collect_image_paths(E.BS_DIR, "BS_oscuras"), E.build_provenance_map(E.HAM_METADATA_PATH), "BS")
    ham["source"] = "ham"; bs["source"] = "bs"
    comb = pd.concat([ham, bs], ignore_index=True)
    dev_i, te_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=E.BASE_SEED).split(comb, groups=comb.lesion_id))
    dev, test = comb.iloc[dev_i].reset_index(drop=True), comb.iloc[te_i].reset_index(drop=True)
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=E.BASE_SEED + 1).split(dev, groups=dev.lesion_id))
    trn, val = dev.iloc[tr_i].reset_index(drop=True), dev.iloc[va_i].reset_index(drop=True)
    assert not set(dev.lesion_id) & set(test.lesion_id)
    return dict(ham_tr=trn[trn.source == "ham"], ham_va=val[val.source == "ham"],
                bs_tr=trn[trn.source == "bs"], bs_va=val[val.source == "bs"],
                bs_test=test[test.source == "bs"].reset_index(drop=True),
                ham_test=test[test.source == "ham"].reset_index(drop=True))


class BestInRAM(keras.callbacks.Callback):
    """Mejor epoca (val_auc) guardada en memoria; evita restaurar desde archivo (problema de EfficientNet-B0)."""
    def __init__(self):
        super().__init__(); self.best = -np.inf; self.weights = None
    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get("val_auc")
        if v is not None and v > self.best:
            self.best = v; self.weights = [w.copy() for w in self.model.get_weights()]


def entrenar(model, tr, va, hist_csv, epochs):
    ram = BestInRAM()
    cbs = [keras.callbacks.EarlyStopping(monitor="val_auc", patience=6, mode="max", restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
           keras.callbacks.CSVLogger(str(hist_csv)), ram]
    model.fit(E.make_dataset(tr, augment=True, shuffle=True),
              validation_data=E.make_dataset(va, augment=False, shuffle=False), epochs=epochs, callbacks=cbs, verbose=2)
    model.set_weights(ram.weights)
    return model


def predecir(model, df):
    y, s = E.predict_scores(model, E.make_dataset(df[["filepath", "label"]], augment=False, shuffle=False))
    return np.asarray(y), np.asarray(s)


def cargar_final(cfg, arch, seed, mod_dir=E.MODELS_DIR):
    """Carga un modelo final archivado. EfficientNet-B0 C40: pesos .npz con cargar_c40_efficientnet.cargar."""
    sfx = "" if seed == E.BASE_SEED else f"_s{seed}"
    if cfg == "C40" and arch == "efficientnetb0":
        from cargar_c40_efficientnet import cargar
        return cargar(Path(mod_dir) / "c40_eff_v2" / f"FINAL_C40_{arch}{sfx}_pesos.npz")
    return keras.models.load_model(Path(mod_dir) / f"FINAL_{cfg}_{arch}{sfx}.keras")
