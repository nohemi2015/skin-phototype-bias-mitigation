"""
p5_verificar_derm7pt_oficial.py - COMPARA LA COPIA DE KAGGLE DE DERM7PT CON LA VERSION OFICIAL (SFU).

Descargar la version oficial (Kawahara et al., https://derm.cs.sfu.ca) y ejecutar:
    python p5_verificar_derm7pt_oficial.py --kaggle "C:\\1. HAM10000\\derm7pt" --oficial "C:\\derm7pt_oficial"
Compara: (1) lista de archivos de imagen por ruta relativa, (2) SHA-256 de cada archivo, (3) si el hash
difiere, tamano y correlacion de pixeles; (4) meta.csv (diagnostico de cada caso).
Si todo coincide, en el articulo basta con decir "verificada archivo por archivo contra la version oficial".
"""
import argparse, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def index(root):
    root = Path(root)
    return {p.relative_to(root).as_posix().lower(): p for p in root.rglob("*") if p.suffix.lower() in EXT}


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kaggle", required=True); ap.add_argument("--oficial", required=True)
    a = ap.parse_args(); K, O = index(a.kaggle), index(a.oficial)
    # las rutas pueden tener un prefijo distinto: emparejar por los dos ultimos componentes
    key = lambda s: "/".join(s.split("/")[-2:])
    K2 = {key(k): v for k, v in K.items()}; O2 = {key(k): v for k, v in O.items()}
    rows = []
    for k in sorted(set(K2) | set(O2)):
        r = dict(imagen=k, en_kaggle=k in K2, en_oficial=k in O2)
        if r["en_kaggle"] and r["en_oficial"]:
            r["sha_igual"] = sha(K2[k]) == sha(O2[k])
            if not r["sha_igual"]:
                x = np.asarray(Image.open(K2[k]).convert("L"), float); y = np.asarray(Image.open(O2[k]).convert("L"), float)
                r["mismo_tamano"] = x.shape == y.shape
                r["correlacion"] = float(np.corrcoef(x.ravel(), y.ravel())[0, 1]) if x.shape == y.shape else np.nan
        rows.append(r)
    d = pd.DataFrame(rows); d.to_csv("derm7pt_kaggle_vs_oficial.csv", index=False)
    print(d[["en_kaggle", "en_oficial"]].value_counts()); print("hash identico:", d.get("sha_igual", pd.Series()).sum())
    mk = next(Path(a.kaggle).rglob("meta.csv"), None); mo = next(Path(a.oficial).rglob("meta.csv"), None)
    if mk and mo:
        A, B = pd.read_csv(mk), pd.read_csv(mo); c = [x for x in A.columns if x in B.columns]
        print("meta.csv identico en columnas comunes:", A[c].equals(B[c]))


if __name__ == "__main__":
    main()
