"""
evaluar_derm7pt.py  (28-oct-2026)
=================================
Evaluacion zero-shot de TODOS los modelos finales (FINAL_*.json) en Derm7pt,
que tiene, para cada lesion, una imagen DERMATOSCOPICA y una CLINICA.
Asi se compara la misma lesion, con la misma etiqueta, en las dos modalidades:
si el AUC cae al pasar de dermatoscopia a foto clinica, la caida se debe a la
modalidad y no a la cohorte, a las etiquetas ni a la mezcla de enfermedades.

Derm7pt: https://derm.cs.sfu.ca  (Kawahara et al., IEEE JBHI 2019). Descomprimir
de modo que exista <derm7pt_dir>/meta/meta.csv y <derm7pt_dir>/images/...

Etiqueta binaria (igual que en HAM10000): maligno = melanoma (todas sus variantes)
y carcinoma basocelular; benigno = nevos, queratosis seborreica, dermatofibroma,
lentigo, lesiones vasculares, melanosis. "miscellaneous" se excluye.

Uso (desde la carpeta del proyecto, donde esta entrenar_leakage_safe_v4.py):
    python evaluar_derm7pt.py --derm7pt_dir "C:\\1. HAM10000\\derm7pt"   (busca meta.csv dentro)
Opcional:
    --ham_dir datos_piel\\pieles_claras   busca imagenes de Derm7pt repetidas en HAM10000
Salida: resultados\\leakage_safe\\derm7pt\\
    predicciones_derm7pt.csv  (modelo, case_num, modalidad, y_true, y_score)
    casos_derm7pt.csv         (casos usados y su etiqueta)
    duplicados_derm7pt_ham.csv (si se da --ham_dir)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from entrenar_leakage_safe_v4 import RESULTS_DIR, make_dataset, predict_scores

MALIGN = ("melanoma", "basal cell carcinoma")


def find_meta(top):
    """Busca meta.csv dentro de la carpeta dada (en cualquier subcarpeta)."""
    top = Path(top)
    direct = top / "meta" / "meta.csv"
    if direct.exists():
        return direct
    cands = [p for p in top.rglob("*.csv") if p.name.lower() == "meta.csv"]
    if not cands:
        cands = [p for p in top.rglob("*.csv")
                 if {"clinic", "derm", "diagnosis"} <= set(pd.read_csv(p, nrows=1).columns)]
    return cands[0] if cands else None


def load_cases(top, meta_path):
    meta = pd.read_csv(meta_path)
    # indice de todas las imagenes bajo la carpeta, para resolver rutas relativas
    files = [p for p in Path(top).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    by_suffix = {}
    for f in files:
        parts = f.parts
        by_suffix.setdefault("/".join(parts[-2:]).lower(), f)
        by_suffix.setdefault(parts[-1].lower(), f)
    def resolve(rel):
        rel = str(rel).replace("\\", "/").strip("/")
        key2 = "/".join(rel.split("/")[-2:]).lower()
        f = by_suffix.get(key2) or by_suffix.get(rel.split("/")[-1].lower())
        return str(f.resolve()) if f is not None else ""
    dx = meta["diagnosis"].astype(str).str.strip().str.lower()
    keep = dx != "miscellaneous"
    meta = meta[keep].copy(); dx = dx[keep]
    meta["label"] = dx.apply(lambda d: int(any(d.startswith(m) for m in MALIGN)))
    meta["derm_path"] = meta["derm"].apply(resolve)
    meta["clinic_path"] = meta["clinic"].apply(resolve)
    ok = (meta["derm_path"] != "") & (meta["clinic_path"] != "")
    if (~ok).any():
        print(f"Aviso: {int((~ok).sum())} casos sin alguna de sus dos imagenes; se excluyen.")
    meta = meta[ok].reset_index(drop=True)
    print(f"meta.csv: {meta_path}  |  imagenes encontradas bajo la carpeta: {len(files)}")
    return meta[["case_num", "diagnosis", "label", "derm_path", "clinic_path"]]


def dhash(path, size=8):
    from PIL import Image
    im = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def check_ham(cases, ham_dir, out):
    ham = [p for p in Path(ham_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"Buscando duplicados: {len(ham)} imagenes de HAM10000 x {2 * len(cases)} de Derm7pt ...")
    H = np.stack([dhash(p) for p in ham])
    rows = []
    for col in ("derm_path", "clinic_path"):
        for c, p in zip(cases.case_num, cases[col]):
            d = np.unpackbits(np.bitwise_xor(H, dhash(p)), axis=1).sum(1)
            j = int(d.argmin())
            if d[j] <= 4:
                rows.append(dict(case_num=c, modality=col[:-5], derm7pt=p, ham=str(ham[j]), hamming=int(d[j])))
    pd.DataFrame(rows, columns=["case_num", "modality", "derm7pt", "ham", "hamming"]).to_csv(out, index=False)
    print(f"Posibles duplicados (Hamming <= 4): {len(rows)}  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derm7pt_dir", required=True)
    ap.add_argument("--ham_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    root = Path(a.derm7pt_dir)
    if not root.exists():
        sys.exit(f"No existe la carpeta {root}. Revisa la ruta DERM7PT en el .bat.")
    meta_path = find_meta(root)
    if meta_path is None:
        print(f"No encontre meta.csv dentro de {root}. Contenido (primeros niveles):")
        for p in sorted(root.rglob("*"))[:40]:
            print("   ", p.relative_to(root))
        sys.exit("Manda esta lista para ajustar el script.")
    out = Path(a.out_dir) if a.out_dir else RESULTS_DIR / "derm7pt"
    out.mkdir(parents=True, exist_ok=True)

    cases = load_cases(root, meta_path)
    cases.to_csv(out / "casos_derm7pt.csv", index=False)
    print(f"Derm7pt: {len(cases)} casos ({int(cases.label.sum())} malignos, {int((cases.label == 0).sum())} benignos)")
    if a.ham_dir:
        check_ham(cases, a.ham_dir, out / "duplicados_derm7pt_ham.csv")

    jsons = sorted(RESULTS_DIR.glob("FINAL_*.json"))
    if not jsons:
        sys.exit(f"No hay modelos finales (FINAL_*.json) en {RESULTS_DIR}")
    pred_file = out / "predicciones_derm7pt.csv"
    done = set(pd.read_csv(pred_file).model) if pred_file.exists() else set()
    for j in jsons:
        if j.stem in done:
            print(f"[ya hecho] {j.stem}"); continue
        info = json.load(open(j))
        model = keras.models.load_model(info["model_file"])
        parts = []
        for mod, col in (("dermoscopic", "derm_path"), ("clinical", "clinic_path")):
            df = pd.DataFrame({"filepath": cases[col].values, "label": cases.label.values})
            y, s = predict_scores(model, make_dataset(df, augment=False, shuffle=False))
            assert (y == cases.label.values).all()
            parts.append(pd.DataFrame({"model": j.stem, "case_num": cases.case_num.values,
                                       "modality": mod, "y_true": y, "y_score": s}))
        res = pd.concat(parts)
        res.to_csv(pred_file, mode="a", header=not pred_file.exists(), index=False)
        from sklearn.metrics import roc_auc_score
        aucs = {m: roc_auc_score(g.y_true, g.y_score) for m, g in res.groupby("modality")}
        print(f"{j.stem}: AUC dermatoscopia {aucs['dermoscopic']:.3f} | clinica {aucs['clinical']:.3f}")
        keras.backend.clear_session()
    print(f"\nListo. Resultados en: {out}")


if __name__ == "__main__":
    main()
