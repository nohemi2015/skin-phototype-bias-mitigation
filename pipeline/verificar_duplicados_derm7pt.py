"""
verificar_duplicados_derm7pt.py  (28-oct-2026)
==============================================
Revisa con una prueba mas fina las parejas Derm7pt-HAM10000 que la busqueda rapida
(dHash 9x8, Hamming <= 4) marco como posibles duplicados.

Para cada pareja calcula la correlacion de las dos imagenes en escala de grises a
128x128 (y con volteos horizontal/vertical; se toma la maxima). Un duplicado real
(misma foto, aunque cambie el tamano) da una correlacion muy alta (> 0.9); dos
lesiones distintas que solo se parecen a baja resolucion dan valores bajos.
Ademas arma laminas con las parejas lado a lado, ordenadas de mayor a menor
correlacion, para revisarlas a ojo.

Uso (desde C:\\1. HAM10000):
    python verificar_duplicados_derm7pt.py
Salida: resultados\\leakage_safe\\derm7pt\\verificacion_duplicados\\
    parejas_correlacion.csv
    lamina_01.png, lamina_02.png, ...
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
SRC = BASE / "resultados" / "leakage_safe" / "derm7pt" / "duplicados_derm7pt_ham.csv"
OUT = SRC.parent / "verificacion_duplicados"
SIDE = 128
POR_LAMINA = 30


def gray(path, n=SIDE):
    return np.asarray(Image.open(path).convert("L").resize((n, n), Image.BILINEAR), dtype=np.float64)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else BASE / p


def main():
    d = pd.read_csv(SRC)
    OUT.mkdir(parents=True, exist_ok=True)
    cs = []
    for _, r in d.iterrows():
        a = gray(resolve(r.derm7pt)); b = gray(resolve(r.ham))
        cs.append(max(corr(a, b), corr(a, b[:, ::-1]), corr(a, b[::-1, :]), corr(a, b[::-1, ::-1])))
    d["correlacion"] = cs
    d = d.sort_values(["correlacion"], ascending=False).reset_index(drop=True)
    d.to_csv(OUT / "parejas_correlacion.csv", index=False)
    print(d.groupby("modality").correlacion.describe(percentiles=[.5, .9, .99]).round(3))
    print("Parejas con correlacion > 0.9:", int((d.correlacion > 0.9).sum()),
          "| > 0.8:", int((d.correlacion > 0.8).sum()))

    # laminas: pareja Derm7pt | HAM10000, con la correlacion y la modalidad
    cols = 3; w = 2 * SIDE + 10; h = SIDE + 18
    for k in range(0, len(d), POR_LAMINA):
        sub = d.iloc[k:k + POR_LAMINA]
        rows = int(np.ceil(len(sub) / cols))
        sheet = Image.new("RGB", (cols * (w + 10), rows * (h + 6)), "white")
        dr = ImageDraw.Draw(sheet)
        for i, (_, r) in enumerate(sub.iterrows()):
            x = (i % cols) * (w + 10); y = (i // cols) * (h + 6)
            for j, p in enumerate((r.derm7pt, r.ham)):
                im = Image.open(resolve(p)).convert("RGB").resize((SIDE, SIDE))
                sheet.paste(im, (x + j * (SIDE + 4), y + 16))
            dr.text((x, y + 2), f"#{k + i + 1} {r.modality} caso {r.case_num}  r={r.correlacion:.2f}", fill="black")
        sheet.save(OUT / f"lamina_{k // POR_LAMINA + 1:02d}.png")
    print(f"Listo: {OUT}")


if __name__ == "__main__":
    main()
