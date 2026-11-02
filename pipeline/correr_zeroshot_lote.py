"""
correr_zeroshot_lote.py  (27-oct-2026)
======================================
Evalua en zero-shot TODOS los modelos finales (resultados/leakage_safe/FINAL_*.json)
en las cuatro cohortes reales, sin pisar resultados anteriores.

Salida: resultados/leakage_safe/zeroshot_lote/<modelo>/<cohorte>/
        (una carpeta por modelo final, p. ej. FINAL_A_mobilenetv2/DDI/)
Si un modelo/cohorte ya tiene su resumen, se salta (se puede volver a lanzar sin miedo).

Uso (desde la carpeta del proyecto):
    python correr_zeroshot_lote.py
"""
import subprocess, sys
from pathlib import Path

from entrenar_leakage_safe_v4 import DATA_DIR, RESULTS_DIR

COHORTES = {                      # nombre -> carpeta con BENIGNAS/ y MALIGNAS/
    "DDI":         DATA_DIR / "DDI_clasificado",
    "DDI2":        DATA_DIR / "DDI2_clasificado",
    "Fitzpatrick": DATA_DIR / "Fitzpatrick_clasificado",
    "PADUFES":     DATA_DIR / "PADUFES_clasificado",
}
OUT_ROOT = RESULTS_DIR / "zeroshot_lote"
EVAL = Path(__file__).resolve().parent / "evaluar_zeroshot_final_v2.py"


def main():
    faltan = [str(p) for p in COHORTES.values()
              if not (p / "BENIGNAS").exists() or not (p / "MALIGNAS").exists()]
    if faltan:
        sys.exit("No encuentro estas carpetas de cohortes (deben tener BENIGNAS y MALIGNAS):\n  "
                 + "\n  ".join(faltan) + "\nCorrige las rutas en COHORTES y vuelve a correr.")
    jsons = sorted(p for p in RESULTS_DIR.glob("FINAL_*.json"))
    if not jsons:
        sys.exit(f"No hay modelos finales (FINAL_*.json) en {RESULTS_DIR}")
    print("Modelos finales encontrados:\n  " + "\n  ".join(j.stem for j in jsons))

    for j in jsons:
        for name, folder in COHORTES.items():
            out = OUT_ROOT / j.stem / name
            if (out / f"resumen_zeroshot_{name}.json").exists():
                print(f"[ya hecho] {j.stem} / {name}"); continue
            print(f"\n=== {j.stem} / {name} ===")
            subprocess.run([sys.executable, str(EVAL), "--final_json", str(j),
                            "--external_dir", str(folder), "--dataset_name", name,
                            "--out_dir", str(out)], check=True)
    print(f"\nListo. Resultados en: {OUT_ROOT}")


if __name__ == "__main__":
    main()
