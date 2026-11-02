"""
listar_excluidas_ham.py  (28-oct-2026)
Lista las imagenes de HAM10000 que estan en HAM10000_metadata.csv pero no en
datos_piel/pieles_claras, e indica cuantas imagenes de Brown Skin derivan de cada una.
Uso (desde la carpeta del proyecto):  python listar_excluidas_ham.py
Salida: ham10000_excluidas.csv
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
meta = pd.read_csv(ROOT / "HAM10000_metadata.csv")
meta["image_id"] = meta["image_id"].str.strip().str.upper()
rx = re.compile(r"ISIC[_]?(\d+)", re.IGNORECASE)
def ids(folder):
    out = []
    for p in (ROOT / "datos_piel" / folder).rglob("*"):
        if p.is_file():
            m = rx.search(p.name)
            if m: out.append("ISIC_" + m.group(1))
    return out
ham = set(ids("pieles_claras"))
bs = pd.Series(ids("pieles_oscuras")).value_counts()
exc = meta[~meta.image_id.isin(ham)].copy()
exc["imagenes_brown_skin_derivadas"] = exc.image_id.map(bs).fillna(0).astype(int)
cols = [c for c in ["image_id", "lesion_id", "dx", "dx_type", "imagenes_brown_skin_derivadas"] if c in exc]
exc[cols].sort_values("image_id").to_csv(ROOT / "ham10000_excluidas.csv", index=False)
print(f"Metadata: {len(meta)} | en pieles_claras: {len(ham)} | excluidas: {len(exc)}")
print(exc["dx"].value_counts().to_string() if "dx" in exc else "")
print(f"Excluidas con derivadas en Brown Skin: {(exc.imagenes_brown_skin_derivadas > 0).sum()}")
print(f"Lesiones con TODAS sus imagenes excluidas: "
      f"{len(set(exc.lesion_id) - set(meta[meta.image_id.isin(ham)].lesion_id))}")
print("Guardado: ham10000_excluidas.csv")
