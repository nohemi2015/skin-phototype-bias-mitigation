"""
cargar_c40_efficientnet.py  (3-nov-2026)
========================================
Carga los pesos de los modelos finales EfficientNet-B0 C40 (FINAL_C40_efficientnetb0*_pesos.npz)
y comprueba que reproducen sus predicciones archivadas del hold-out de Brown Skin.

Por que hace falta: en EfficientNet-B0 con las ultimas 40 capas descongeladas, el orden de las
variables del modelo ajustado (construido con clone_model a partir del modelo fuente A) no es el
mismo que el de un EfficientNet-B0 construido desde cero. Restaurar estos pesos en un modelo con
otro orden (por ejemplo con keras.models.load_model de un .keras) asigna valores a variables
equivocadas, sobre todo en las capas de batch normalization, y el modelo pierde discriminacion
sin dejar de funcionar. Por eso los pesos se publican como arrays en el orden del modelo ajustado,
y este script reconstruye exactamente ese modelo (build_model -> clone_model -> ultimas 40 capas
entrenables) antes de asignarlos.

Uso (desde la carpeta del proyecto):
    python cargar_c40_efficientnet.py --pesos "modelos\\leakage_safe\\c40_eff_v2\\FINAL_C40_efficientnetb0_pesos.npz" ^
        --pred "resultados\\leakage_safe\\c40_eff_v2\\pred_FINAL_C40_efficientnetb0_test_BS.csv"
Si se omite --pred solo carga el modelo y lo guarda en memoria para usarlo desde otro script
(funcion cargar()).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tensorflow import keras

import entrenar_leakage_safe_v4 as E


def cargar(pesos_npz, arch="efficientnetb0", n_layers=40):
    base = E.build_model(arch, train_base=False)
    m = keras.models.clone_model(base); m.set_weights(base.get_weights())
    m = E.unfreeze_last_layers(m, n_layers=n_layers)
    z = np.load(pesos_npz)
    m.set_weights([z[f"arr_{i}"] for i in range(len(z.files))])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", required=True)
    ap.add_argument("--pred", default=None, help="pred_FINAL_*_test_BS.csv archivado")
    a = ap.parse_args()
    m = cargar(a.pesos)
    print("Modelo cargado:", a.pesos)
    if a.pred:
        p = pd.read_csv(a.pred)
        y, s = E.predict_scores(m, E.make_dataset(pd.DataFrame({"filepath": p.filepath, "label": p.y_true}),
                                                  augment=False, shuffle=False))
        print(f"AUC archivado {roc_auc_score(p.y_true, p.y_score):.4f} | AUC recalculado {roc_auc_score(y, s):.4f} "
              f"| max |diferencia| {np.abs(s - p.y_score.values).max():.2e}")


if __name__ == "__main__":
    main()
