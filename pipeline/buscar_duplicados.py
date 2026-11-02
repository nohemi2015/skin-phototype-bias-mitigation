"""
buscar_duplicados.py  (28-oct-2026)
Busca imagenes duplicadas entre las cohortes externas (DDI, DDI-2, Fitzpatrick17k,
PAD-UFES-20): duplicados exactos (mismo MD5) y casi duplicados (hash perceptual de
diferencia, 64 bits, distancia de Hamming <= 4 tras redimensionar).
Uso (desde la carpeta del proyecto):  python buscar_duplicados.py
Salida: duplicados_entre_cohortes.csv
"""
import hashlib, itertools
from pathlib import Path
import numpy as np, pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent / "datos_piel"
COH = {"DDI": "DDI_clasificado", "DDI2": "DDI2_clasificado",
       "Fitzpatrick": "Fitzpatrick_clasificado", "PADUFES": "PADUFES_clasificado"}
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
rows = []
for name, folder in COH.items():
    for p in (ROOT / folder).rglob("*"):
        if not (p.is_file() and p.suffix.lower() in EXTS): continue
        raw = p.read_bytes()
        try:
            img = tf.image.decode_image(raw, channels=1, expand_animations=False)
            small = tf.image.resize(tf.cast(img, tf.float32), (8, 9)).numpy()[..., 0]
            dh = (small[:, 1:] > small[:, :-1]).flatten()
        except Exception:
            dh = None
        rows.append({"cohorte": name, "archivo": p.name, "md5": hashlib.md5(raw).hexdigest(), "dhash": dh})
d = pd.DataFrame(rows); print(d.cohorte.value_counts().to_string())
out = []
for md5, g in d.groupby("md5"):
    if g.cohorte.nunique() > 1 or len(g) > 1:
        for a, b in itertools.combinations(g.index, 2):
            out.append({"tipo": "exacto", "cohorte_1": d.cohorte[a], "archivo_1": d.archivo[a],
                        "cohorte_2": d.cohorte[b], "archivo_2": d.archivo[b], "distancia": 0})
ok = d[d.dhash.notna()].reset_index(drop=True)
H = np.stack(ok.dhash.values).astype(np.uint8)
for i in range(len(ok)):
    dist = (H[i + 1:] != H[i]).sum(1)
    for j in np.where(dist <= 4)[0] + i + 1:
        if ok.md5[i] == ok.md5[j]: continue
        out.append({"tipo": "casi", "cohorte_1": ok.cohorte[i], "archivo_1": ok.archivo[i],
                    "cohorte_2": ok.cohorte[j], "archivo_2": ok.archivo[j], "distancia": int(dist[j - i - 1])})
r = pd.DataFrame(out, columns=["tipo", "cohorte_1", "archivo_1", "cohorte_2", "archivo_2", "distancia"])
r.to_csv("duplicados_entre_cohortes.csv", index=False)
if len(r):
    r["par"] = r.cohorte_1 + "-" + r.cohorte_2
    print(r.groupby(["tipo", "par"]).size().to_string())
else:
    print("Sin duplicados")
print("Guardado: duplicados_entre_cohortes.csv")
