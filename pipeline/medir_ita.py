"""
medir_ita.py  (28-oct-2026)
===========================
Mide el tono de piel de cada imagen con el angulo de tipologia individual (ITA),
    ITA = arctan((L* - 50) / b*) * 180 / pi      (CIELAB, iluminante D65),
calculado sobre la piel que rodea a la lesion. Sirve para comprobar si Brown Skin
tiene realmente tonos de piel oscura comparables a los de las cohortes reales.

Piel = pixeles que no son lesion ni artefacto: se descartan los pixeles mas oscuros
de la imagen (lesion, pelo; por debajo del percentil 40 de L*), los muy oscuros
(vineta del dermatoscopio, L* < 15) y los brillos (L* > 95). El ITA de la imagen
es la mediana de los pixeles restantes. Es una estimacion aproximada, igual para
todas las bases, y se valida con los grupos de fototipo de DDI (deberia bajar de
I-II a V-VI).

Uso (desde la carpeta del proyecto):
    python medir_ita.py
    python medir_ita.py --solo_derm7pt   (misma lesion en dermatoscopia y en foto clinica:
                                         mide cuanto cambia el ITA solo por la modalidad)
    python medir_ita.py --carpeta "C:\\ruta\\imagenes_de_estilo"   (cualquier carpeta)
Opcional: --max_por_base 1500 (imagenes por base, al azar; por defecto todas)
Salida: resultados\\leakage_safe\\ita\\ita_por_imagen.csv
"""
import argparse, re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from entrenar_leakage_safe_v4 import DATA_DIR, RESULTS_DIR, PROJECT_DIR

BASES = {
    "HAM10000":    DATA_DIR / "pieles_claras",
    "BrownSkin":   DATA_DIR / "pieles_oscuras",
    "DDI":         DATA_DIR / "DDI_clasificado",
    "DDI2":        DATA_DIR / "DDI2_clasificado",
    "Fitzpatrick": DATA_DIR / "Fitzpatrick_clasificado",
    "PADUFES":     DATA_DIR / "PADUFES_clasificado",
}
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def srgb_to_lab(rgb):
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    b = 200 * (f[..., 1] - f[..., 2])
    return L, b


def ita_image(path, size=256):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    L, b = srgb_to_lab(np.asarray(im, dtype=np.float64))
    L, b = L.ravel(), b.ravel()
    m = (L > 15) & (L < 95)
    if m.sum() < 100:
        return np.nan, np.nan
    L, b = L[m], b[m]
    keep = L >= np.percentile(L, 40)
    ita = np.degrees(np.arctan2(L[keep] - 50, b[keep]))
    return float(np.median(ita)), float(np.median(L[keep]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_por_base", type=int, default=0)
    ap.add_argument("--solo_derm7pt", action="store_true",
                    help="mide solo Derm7pt (dermatoscopica y clinica de cada lesion)")
    ap.add_argument("--carpeta", default=None,
                    help="mide todas las imagenes de una carpeta cualquiera (p. ej., las imagenes de estilo)")
    a = ap.parse_args()
    out = RESULTS_DIR / "ita"; out.mkdir(parents=True, exist_ok=True)
    if a.carpeta:
        files = sorted(p for p in Path(a.carpeta).rglob("*") if p.suffix.lower() in EXTS)
        rows = []
        for p in files:
            ita, L = ita_image(p)
            rows.append(dict(archivo=str(p.relative_to(a.carpeta)), ita=ita, L=L))
        d = pd.DataFrame(rows); d.to_csv(out / "ita_carpeta.csv", index=False)
        pd.set_option("display.max_rows", 200)
        print(d.round(1).to_string(index=False))
        print(f"Listo: {out / 'ita_carpeta.csv'}")
        return
    if a.solo_derm7pt:
        casos = pd.read_csv(RESULTS_DIR / "derm7pt" / "casos_derm7pt.csv")
        rows = []
        for i, r in casos.iterrows():
            for mod, col in (("dermoscopic", "derm_path"), ("clinical", "clinic_path")):
                ita, L = ita_image(r[col])
                rows.append(dict(case_num=r.case_num, modality=mod, label=r.label, ita=ita, L=L))
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(casos)}")
        d = pd.DataFrame(rows); d.to_csv(out / "ita_derm7pt.csv", index=False)
        w = d.pivot(index="case_num", columns="modality", values="ita")
        print(d.groupby("modality").ita.describe().round(1))
        print("Diferencia pareada (dermatoscopica - clinica), mediana:",
              round(float((w.dermoscopic - w.clinical).median()), 1))
        print(f"Listo: {out / 'ita_derm7pt.csv'}")
        return
    ddi_meta = PROJECT_DIR / "ddi_metadata.csv"
    ddi = None
    if ddi_meta.exists():
        d = pd.read_csv(ddi_meta)
        ddi = dict(zip(d["DDI_file"].astype(str), d["skin_tone"].astype(str)))
    rows = []
    for name, folder in BASES.items():
        if not folder.exists():
            print(f"[no existe, se omite] {folder}"); continue
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)
        if a.max_por_base and len(files) > a.max_por_base:
            rng = np.random.default_rng(2026)
            files = [files[i] for i in sorted(rng.choice(len(files), a.max_por_base, replace=False))]
        print(f"{name}: {len(files)} imagenes")
        for i, p in enumerate(files):
            try:
                ita, L = ita_image(p)
            except Exception as e:
                ita, L = np.nan, np.nan
            grupo = ""
            if name == "DDI" and ddi is not None:
                grupo = ddi.get(p.name, "")
            if name == "BrownSkin":
                m = re.search(r"fototipo(\w+)", p.stem, re.I)
                grupo = m.group(1) if m else ""
            rows.append(dict(base=name, archivo=p.name, clase=p.parent.name, grupo=grupo, ita=ita, L=L))
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(files)}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "ita_por_imagen.csv", index=False)
    print(df.groupby(["base", "grupo"]).ita.describe(percentiles=[.25, .5, .75]).round(1))
    print(f"\nListo: {out / 'ita_por_imagen.csv'}")


if __name__ == "__main__":
    main()
