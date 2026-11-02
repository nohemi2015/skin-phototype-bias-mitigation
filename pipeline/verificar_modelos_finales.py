"""
verificar_modelos_finales.py  (28-oct-2026)
===========================================
Auditoria de los modelos finales (sin entrenar nada):
 1) Recarga cada FINAL_*.keras, vuelve a predecir las imagenes Brown Skin del hold-out
    y compara con las predicciones archivadas (pred_FINAL_*_test_BS.csv).
 2) Describe la arquitectura guardada (capa Rescaling, capas entrenables del backbone).
 3) Resume las curvas de entrenamiento (hist_*.csv): mejor val_auc y epocas, para los
    modelos finales y para los C40 de la validacion cruzada.
Uso (desde la carpeta del proyecto):  python verificar_modelos_finales.py
Salida: resultados/leakage_safe/verificacion/  (+ verificacion_modelos.zip para enviar)
"""
import json, shutil, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from entrenar_leakage_safe_v4 import RESULTS_DIR, PRED_DIR, make_dataset, predict_scores

OUT = RESULTS_DIR / "verificacion"; OUT.mkdir(parents=True, exist_ok=True)
rows = []
for j in sorted(RESULTS_DIR.glob("FINAL_*.json")):
    info = json.load(open(j)); tag = j.stem[len("FINAL_"):]
    r = {"modelo": j.stem, "control_auc_json": info.get("control_bs_auc"),
         "umbral": info.get("frozen_threshold_from_bs_val"), "semilla": info.get("train_seed", 2026)}
    pf = PRED_DIR / f"pred_FINAL_{tag}_test_BS.csv"
    try:
        m = keras.models.load_model(info["model_file"])
        r["rescaling"] = any(isinstance(l, keras.layers.Rescaling) for l in m.layers)
        base = [l for l in m.layers if isinstance(l, keras.Model)]
        if base:
            r["capas_backbone"] = len(base[0].layers)
            r["capas_backbone_entrenables"] = sum(l.trainable for l in base[0].layers)
        if pf.exists():
            p = pd.read_csv(pf)
            df = pd.DataFrame({"filepath": p["filepath"], "label": p["y_true"]})
            y, s = predict_scores(m, make_dataset(df, augment=False, shuffle=False))
            r["n_holdout"] = len(y)
            r["auc_recalculado"] = roc_auc_score(y, s)
            r["auc_archivo"] = roc_auc_score(p["y_true"], p["y_score"])
            r["max_dif_scores"] = float(np.max(np.abs(s - p["y_score"].values)))
            shutil.copy(pf, OUT / pf.name)
        else:
            r["nota"] = "sin archivo de predicciones del hold-out"
        keras.backend.clear_session()
    except Exception as e:
        r["error"] = repr(e)
    rows.append(r); print(r)
pd.DataFrame(rows).to_csv(OUT / "verificacion_modelos_finales.csv", index=False)

hist = []
for h in sorted(RESULTS_DIR.glob("hist_*.csv")):
    if not (h.stem.startswith("hist_FINAL_") or "C40" in h.stem or h.stem.startswith("hist_A_")):
        continue
    d = pd.read_csv(h)
    if "val_auc" not in d: continue
    hist.append({"archivo": h.stem, "epocas": len(d), "mejor_val_auc": d.val_auc.max(),
                 "epoca_mejor": int(d.val_auc.idxmax()) + 1, "val_auc_final": d.val_auc.iloc[-1]})
    shutil.copy(h, OUT / h.name)
pd.DataFrame(hist).to_csv(OUT / "resumen_historiales.csv", index=False)
print(pd.DataFrame(hist).to_string())

z = Path("verificacion_modelos.zip")
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
    for x in OUT.iterdir(): f.write(x, x.name)
print(f"\nListo. Envia {z.resolve()}")
