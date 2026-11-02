"""
buscar_ham_faltantes.py  (28-oct-2026)
Busca en tu computadora los archivos originales de las 343 imagenes de HAM10000
que faltan en datos_piel/pieles_claras (lista en ham10000_excluidas.csv) y prueba
si se pueden abrir. Solo lee archivos; no copia ni borra nada.
Uso:  python buscar_ham_faltantes.py  [carpeta1 carpeta2 ...]
      (sin argumentos busca en esta carpeta, Descargas, Documentos y Escritorio)
Salida: ham10000_faltantes_busqueda.csv
"""
import os, sys
from pathlib import Path
import pandas as pd
import tensorflow as tf

here = Path(__file__).resolve().parent
ids = set(pd.read_csv(here / "ham10000_excluidas.csv")["image_id"].str.upper())
home = Path(os.environ.get("USERPROFILE", Path.home()))
roots = [Path(a) for a in sys.argv[1:]] or [here, home / "Downloads", home / "Documents", home / "Desktop",
                                            home / "Descargas", home / "Documentos", home / "Escritorio"]
skip = (here / "datos_piel" / "pieles_claras").resolve()
rows = []
for root in roots:
    if not root.exists(): continue
    print("Buscando en", root)
    for dp, _, files in os.walk(root):
        if Path(dp).resolve() == skip: continue
        for f in files:
            stem = Path(f).stem.upper()
            if stem in ids:
                p = Path(dp) / f
                try:
                    img = tf.image.decode_image(tf.io.read_file(str(p)), channels=3, expand_animations=False)
                    ok, shape = True, tuple(img.shape)
                except Exception as e:
                    ok, shape = False, repr(e)[:80]
                rows.append({"image_id": stem, "ruta": str(p), "se_abre": ok, "detalle": shape,
                             "bytes": p.stat().st_size})
d = pd.DataFrame(rows, columns=["image_id", "ruta", "se_abre", "detalle", "bytes"])
d.to_csv(here / "ham10000_faltantes_busqueda.csv", index=False)
print(f"\nEncontradas: {d.image_id.nunique()} de {len(ids)} | se abren: {d[d.se_abre].image_id.nunique()}")
print("Guardado: ham10000_faltantes_busqueda.csv")
