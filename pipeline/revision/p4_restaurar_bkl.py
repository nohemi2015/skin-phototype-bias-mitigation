"""
p4_restaurar_bkl.py - RESTAURA LAS 343 IMAGENES BKL QUE FALTAN EN pieles_claras (revision, Major 4).

Copia (no mueve) las imagenes listadas en results/verification/ham10000_excluidas.csv desde la
carpeta original de HAM10000 a datos_piel/pieles_claras/BENIGNAS y comprueba que cada una decodifica.
Despues hay que REPETIR TODO el pipeline (validacion cruzada y modelos finales): al cambiar el
conjunto de imagenes cambian las particiones generadas por StratifiedGroupKFold y GroupShuffleSplit,
asi que no basta con reentrenar los modelos finales. Conservar la version actual de resultados para
reportar ambas (version con 9,672 y con 10,015 imagenes) en el material suplementario.

Uso:  python p4_restaurar_bkl.py --origen "C:\\ruta\\HAM10000_images"  [--lista results\\verification\\ham10000_excluidas.csv]
"""
import argparse, shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import entrenar_leakage_safe_v4 as E


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--origen", required=True)
    ap.add_argument("--lista", default=str(E.PROJECT_DIR / "results" / "verification" / "ham10000_excluidas.csv"))
    a = ap.parse_args()
    ids = pd.read_csv(a.lista).image_id.str.strip().tolist(); dest = E.HAM_DIR / "BENIGNAS"; dest.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(E.HAM_METADATA_PATH).set_index("image_id")
    assert (meta.loc[ids, "dx"] == "bkl").all(), "alguna imagen de la lista no es BKL"
    falt = []
    for i in ids:
        src = next(Path(a.origen).rglob(f"{i}.jpg"), None)
        if src is None: falt.append(i); continue
        Image.open(src).convert("RGB")                      # comprueba que decodifica
        if not (dest / src.name).exists(): shutil.copy2(src, dest / src.name)
    print(f"copiadas/presentes: {len(ids) - len(falt)} de {len(ids)}; no encontradas: {len(falt)}")
    print("Siguiente paso: repetir correr_fase2.bat, correr_fase4_cv_mobilenet.bat, correr_c40_eff.bat,")
    print("correr_semillas_AB.bat, correr_fase5_fusion.bat y correr_fase4_derm7pt_ita.bat con LS_OUT apuntando")
    print("a una carpeta nueva (por ejemplo set LS_OUT=C:\\1. HAM10000\\v4_con_bkl) para no sobrescribir la version actual.")


if __name__ == "__main__":
    main()
