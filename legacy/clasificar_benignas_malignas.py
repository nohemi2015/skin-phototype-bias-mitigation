r"""
clasificar_benignas_malignas.py

Organiza una carpeta plana de imagenes en subcarpetas BENIGNAS/ y
MALIGNAS/ segun la etiqueta de un archivo de metadatos (xlsx o csv).

Asume que el nombre del archivo de imagen (sin extension) coincide con
el identificador de la columna de ID en la planilla (p.ej. photo_id
"502948221_cropped" -> archivo "502948221_cropped.jpg").

Uso:
    python clasificar_benignas_malignas.py ^
        --metadata final_DDI2_Asian_spreadsheet.xlsx ^
        --images_dir "C:\ruta\a\imagenes_sueltas" ^
        --out_dir "C:\ruta\a\dataset_clasificado" ^
        --id_column photo_id ^
        --label_column "benign/malignant"

Por defecto COPIA los archivos (no los mueve), para no arriesgar tus
originales. Usa --mode move si de verdad quieres moverlos.
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Normaliza distintas formas de escribir benigno/maligno hacia los
# nombres de carpeta que ya usas en el resto del proyecto.
# Incluye tanto el formato de texto (DDI2 Asian: "benign"/"malignant")
# como el booleano del DDI original (columna "malignant": True/False).
LABEL_TO_FOLDER = {
    "benign": "BENIGNAS",
    "benigno": "BENIGNAS",
    "false": "BENIGNAS",
    "0": "BENIGNAS",
    "malignant": "MALIGNAS",
    "maligno": "MALIGNAS",
    "true": "MALIGNAS",
    "1": "MALIGNAS",
}


def normalize_id(value):
    """502948221 (float/int leido de excel) -> '502948221'; strings tal cual.
    Si el valor ya incluye extension (p.ej. DDI_file='000001.png' en el
    DDI original), se la quita para poder comparar contra el stem del
    archivo real en disco."""
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int,)):
        return str(value)
    text = str(value).strip()
    stem, ext = os.path.splitext(text)
    if ext.lower() in IMG_EXTENSIONS:
        return stem
    return text


def load_metadata(path, id_column, label_column, malignant_values=None):
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if id_column not in df.columns:
        raise ValueError(f"Columna de ID '{id_column}' no existe. Columnas disponibles: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Columna de etiqueta '{label_column}' no existe. Columnas disponibles: {list(df.columns)}")

    malignant_set = None
    if malignant_values:
        malignant_set = {v.strip().lower() for v in malignant_values.split(",")}

    id_to_label = {}
    for _, row in df.iterrows():
        key = normalize_id(row[id_column])
        if key is None:
            continue
        label_raw = str(row[label_column]).strip().lower()

        if malignant_set is not None:
            # Modo multi-clase: cualquier valor en --malignant_values es
            # maligno, todo lo demas (incluido NaN/otros codigos) es benigno.
            label_raw = "malignant" if label_raw in malignant_set else "benign"

        id_to_label[key] = label_raw

    return id_to_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Ruta al xlsx/csv con las etiquetas")
    ap.add_argument("--images_dir", required=True, help="Carpeta con las imagenes sueltas")
    ap.add_argument("--out_dir", required=True, help="Carpeta destino (se crean BENIGNAS/ y MALIGNAS/ dentro)")
    ap.add_argument("--id_column", default="photo_id")
    ap.add_argument("--label_column", default="benign/malignant")
    ap.add_argument(
        "--malignant_values", default=None,
        help="Opcional: lista separada por comas de valores de la columna de "
             "etiqueta que cuentan como malignos (para datasets multi-clase "
             "como PAD-UFES-20, ej: 'BCC,SCC,MEL'). Todo lo demas se trata "
             "como benigno."
    )
    ap.add_argument("--mode", choices=["copy", "move"], default="copy")
    args = ap.parse_args()

    print(f"Leyendo metadatos: {args.metadata}")
    id_to_label = load_metadata(args.metadata, args.id_column, args.label_column, args.malignant_values)
    print(f"Registros en la planilla: {len(id_to_label)}")

    unknown_labels = {v for v in id_to_label.values() if v not in LABEL_TO_FOLDER}
    if unknown_labels:
        print(f"[aviso] etiquetas no reconocidas (se omitiran esas filas): {unknown_labels}")

    for folder in LABEL_TO_FOLDER.values():
        os.makedirs(os.path.join(args.out_dir, folder), exist_ok=True)

    images_dir = Path(args.images_dir)
    image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    print(f"Imagenes encontradas en {images_dir}: {len(image_files)}")

    matched_ids = set()
    unmatched_images = []
    action = shutil.copy2 if args.mode == "copy" else shutil.move

    for img_path in image_files:
        stem = img_path.stem
        label = id_to_label.get(stem)

        if label is None or label not in LABEL_TO_FOLDER:
            unmatched_images.append(img_path.name)
            continue

        dest_folder = os.path.join(args.out_dir, LABEL_TO_FOLDER[label])
        dest_path = os.path.join(dest_folder, img_path.name)
        action(str(img_path), dest_path)
        matched_ids.add(stem)

    unmatched_metadata = [k for k in id_to_label if k not in matched_ids]

    print(f"\n{'Copiadas' if args.mode == 'copy' else 'Movidas'}: {len(matched_ids)}")
    print(f"Imagenes en la carpeta SIN entrada en la planilla: {len(unmatched_images)}")
    print(f"Filas de la planilla SIN imagen correspondiente: {len(unmatched_metadata)}")

    if unmatched_images:
        report_path = os.path.join(args.out_dir, "imagenes_sin_metadata.csv")
        pd.DataFrame({"filename": unmatched_images}).to_csv(report_path, index=False)
        print(f"  -> detalle guardado en: {report_path}")

    if unmatched_metadata:
        report_path = os.path.join(args.out_dir, "metadata_sin_imagen.csv")
        pd.DataFrame({"id": unmatched_metadata}).to_csv(report_path, index=False)
        print(f"  -> detalle guardado en: {report_path}")

    print(f"\nListo. Revisa: {args.out_dir}")


if __name__ == "__main__":
    main()
