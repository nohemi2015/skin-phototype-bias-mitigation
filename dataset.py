from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    BATCH_SIZE,
    BS_DIR,
    CLASS_NAMES,
    HAM_DIR,
    IMAGE_EXTS,
    IMG_SIZE,
    PARTITIONS_DIR,
    SEED,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)


def collect_image_paths(root_dir, domain_name):
    root_dir = Path(root_dir)
    rows = []

    if not root_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {root_dir}")

    for class_name, label in CLASS_NAMES.items():
        class_dir = root_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"No existe {class_dir}. "
                f"Esperaba subcarpetas: {list(CLASS_NAMES.keys())}"
            )

        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append(
                    {
                        "filepath": str(path.resolve()),
                        "label": int(label),
                        "class_name": class_name,
                        "domain": domain_name,
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No se encontraron imágenes en {root_dir}")

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def load_dataframes():
    ham_df = collect_image_paths(HAM_DIR, "HAM_claras")
    bs_df = collect_image_paths(BS_DIR, "BS_oscuras")
    return ham_df, bs_df


def summarize_df(df, name):
    print(f"\n{name}: {len(df)} imágenes")
    print(df.groupby(["domain", "class_name", "label"]).size())


def _validate_split_ratios():
    total = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "TRAIN_SIZE + VAL_SIZE + TEST_SIZE debe sumar 1.0. "
            f"Valor actual: {total}"
        )


def split_dataframe_fixed(df):
    """Crea una división fija 70/15/15, estratificada por clase."""
    _validate_split_ratios()

    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_SIZE + TEST_SIZE),
        stratify=df["label"],
        random_state=SEED,
    )

    relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=SEED,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_fixed_partitions(df, prefix, overwrite=False):
    """Guarda train, val y test en CSV y evita sobrescribirlos por accidente."""
    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": PARTITIONS_DIR / f"{prefix}_train.csv",
        "val": PARTITIONS_DIR / f"{prefix}_val.csv",
        "test": PARTITIONS_DIR / f"{prefix}_test.csv",
    }

    if not overwrite and all(path.exists() for path in paths.values()):
        raise FileExistsError(
            f"Las particiones de {prefix} ya existen en {PARTITIONS_DIR}. "
            "No se sobrescribieron. Usa overwrite=True solo si deseas regenerarlas."
        )

    train_df, val_df, test_df = split_dataframe_fixed(df)

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    return train_df, val_df, test_df


def load_fixed_partitions(prefix):
    """Carga las particiones previamente guardadas."""
    paths = {
        "train": PARTITIONS_DIR / f"{prefix}_train.csv",
        "val": PARTITIONS_DIR / f"{prefix}_val.csv",
        "test": PARTITIONS_DIR / f"{prefix}_test.csv",
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos de partición:\n- " + "\n- ".join(missing)
        )

    return (
        pd.read_csv(paths["train"]),
        pd.read_csv(paths["val"]),
        pd.read_csv(paths["test"]),
    )


def summarize_partitions(train_df, val_df, test_df, name):
    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    for split_name, split_df in [
        ("Entrenamiento", train_df),
        ("Validación", val_df),
        ("Prueba", test_df),
    ]:
        counts = split_df["class_name"].value_counts().sort_index()
        print(f"\n{split_name}: {len(split_df)} imágenes")
        print(counts.to_string())


def decode_image(filepath, label, augment=False):
    img_bytes = tf.io.read_file(filepath)
    img = tf.image.decode_image(
        img_bytes,
        channels=3,
        expand_animations=False,
    )
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.15)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)

    return img, tf.cast(label, tf.float32)


def make_dataset(df, augment=False, shuffle=True, batch_size=BATCH_SIZE):
    paths = df["filepath"].values
    labels = df["label"].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda x, y: decode_image(x, y, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
