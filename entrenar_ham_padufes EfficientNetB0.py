"""
entrenar_ham_padufes.py

Igual que el pipeline original (HAM10000 -> Brown Skin), pero
reemplazando Brown Skin (sintetico) por PAD-UFES-20 (real) como dominio
objetivo. Preentrena/evalua A sobre HAM10000, y adapta hacia PAD-UFES-20 en
las configuraciones B, C10-C80, D, E.

Ejemplos:
    python entrenar_ham_padufes.py --model C40
    python entrenar_ham_padufes.py --model A --repeat 1

Modelos admitidos:
    A, B, C10, C20, C40, C60, C80, D, E

Nota metodologica: a diferencia de Brown Skin, PAD-UFES-20 es un dataset real
independiente de HAM10000 (no un derivado por transferencia de estilo),
asi que aqui NO aplica el control de leakage por base_lesion_id que se
necesito para Brown Skin -- la particion aleatoria estratificada de PAD-UFES-20
es adecuada tal cual.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from config import HAM_DIR, PADUFES_DIR, MODELS_DIR, RESULTS_DIR, EPOCHS_BASE, EPOCHS_FT
from dataset import collect_image_paths, summarize_df, make_dataset
from models import build_mobilenetv2, build_efficientnetb0, unfreeze_last_layers
from evaluation import predict_scores, evaluate_predictions

MODEL_BUILDERS = {
    "mobilenetv2": build_mobilenetv2,
    "efficientnetb0": build_efficientnetb0,
}

N_SPLITS = 5
N_REPEATS = 3
BASE_SEED = 2026

REPRO_MODELS_DIR = MODELS_DIR / "repetibilidad_padufes"
REPRO_RESULTS_DIR = RESULTS_DIR / "repetibilidad_padufes"
RESULTS_CSV = REPRO_RESULTS_DIR / "resultados_repetibilidad_padufes.csv"

REPRO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPRO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_load_model(path):
    """Intenta cargar un modelo guardado; si el archivo esta corrupto o
    incompleto (por ejemplo, porque el proceso se interrumpio a medio
    checkpoint), lo borra y devuelve None para que se reentrene en vez
    de fallar con un traceback."""
    try:
        return keras.models.load_model(path)
    except Exception as e:
        print(f"⚠️ Modelo en {path} no se pudo cargar ({type(e).__name__}: {e}); "
              f"se borra y se reentrena.")
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def callbacks(model_path):
    history_path = REPRO_RESULTS_DIR / f"historial_{model_path.stem}.csv"
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=6, mode="max",
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_auc",
            mode="max", save_best_only=True, verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=str(history_path), append=False
        ),
    ]


def load_done_keys():
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        return set()

    df = pd.read_csv(RESULTS_CSV)
    required = {"backbone", "model", "repeat", "fold"}
    if df.empty or not required.issubset(df.columns):
        return set()

    return {
        (str(row["backbone"]), str(row["model"]), int(row["repeat"]), int(row["fold"]))
        for _, row in df.iterrows()
    }


def extract_done_a(done):
    """De todas las (backbone, model, repeat, fold) completadas, devuelve
    solo las de A como (backbone, repeat, fold), para saber en que folds
    el modelo A ya termino de verdad (y no es un checkpoint parcial)."""
    return {(b, r, f) for (b, m, r, f) in done if m == "A"}


def save_incremental(metrics):
    """Agrega una fila al CSV acumulado, tolerando que distintos modelos
    tengan columnas distintas (p.ej. 'unfreeze_layers' solo existe para
    C10-C80). Reescribe el archivo completo alineando columnas por
    union, en vez de usar append con encabezado fijo, para evitar
    corromper el CSV si se corren modelos en distinto orden."""
    new_row = pd.DataFrame([metrics])

    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0:
        existing = pd.read_csv(RESULTS_CSV)
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        combined = new_row

    combined.to_csv(RESULTS_CSV, index=False)

    print(
        f"\n✅ Guardado: {metrics['backbone']} | {metrics['model']} | "
        f"repetición {metrics['repeat']} | fold {metrics['fold']} | "
        f"AUC={metrics['auc']:.4f}"
    )


def make_repeated_splits(df):
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=BASE_SEED,
    )

    splits = []
    for index, (train_idx, val_idx) in enumerate(
        splitter.split(df, df["label"])
    ):
        repeat = index // N_SPLITS + 1
        fold = index % N_SPLITS + 1
        splits.append((repeat, fold, train_idx, val_idx))
    return splits


def make_ham_splits(df):
    all_splits = {}

    for repeat in range(1, N_REPEATS + 1):
        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=BASE_SEED + repeat,
        )

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(df, df["label"]), start=1
        ):
            all_splits[(repeat, fold)] = (train_idx, val_idx)

    return all_splits


def evaluate_and_save(
    model, val_df, model_name, repeat, fold,
    model_path, seed, extra=None
):
    val_ds = make_dataset(val_df, augment=False, shuffle=False)
    y_true, y_score = predict_scores(model, val_ds)
    metrics = evaluate_predictions(y_true, y_score)

    metrics.update({
        "model": model_name,
        "repeat": repeat,
        "fold": fold,
        "seed": seed,
        "model_file": str(model_path),
        "n_val": len(val_df),
    })

    if extra:
        metrics.update(extra)

    save_incremental(metrics)


def train_model(model, train_df, val_df, model_path, epochs):
    train_ds = make_dataset(train_df, augment=True, shuffle=True)
    val_ds = make_dataset(val_df, augment=False, shuffle=False)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks(model_path),
        verbose=1,
    )

    return keras.models.load_model(model_path)


def ensure_a_model(backbone, ham_train, ham_val, repeat, fold, done_a):
    model_path = REPRO_MODELS_DIR / f"A_{backbone}_cv_rep{repeat}_fold{fold}.keras"

    if (backbone, repeat, fold) in done_a and model_path.exists():
        model = safe_load_model(model_path)
        if model is not None:
            print(f"🔁 Cargando A existente: {model_path.name}")
            return model, model_path

    if model_path.exists():
        print(f"⚠️ Checkpoint parcial encontrado para A (no está en el CSV de "
              f"resultados, probablemente de una corrida interrumpida) — "
              f"se borra y se reentrena: {model_path.name}")
        model_path.unlink()

    print(f"🚀 Entrenando A [{backbone}] | repetición {repeat} | fold {fold}")
    model = MODEL_BUILDERS[backbone](train_base=False)
    model = train_model(
        model, ham_train, ham_val, model_path, EPOCHS_BASE
    )
    return model, model_path


def run_configuration(backbone, requested_model, requested_repeat=None):
    backbone = backbone.lower()
    if backbone not in MODEL_BUILDERS:
        raise ValueError(f"Backbone no válido: {backbone}. Opciones: {list(MODEL_BUILDERS)}")

    requested_model = requested_model.upper()

    valid_models = {
        "A", "B", "C10", "C20", "C40",
        "C60", "C80", "D", "E"
    }

    if requested_model not in valid_models:
        raise ValueError(
            f"Modelo no válido: {requested_model}. "
            f"Opciones: {sorted(valid_models)}"
        )

    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    print(
        f"Protocolo: {N_REPEATS} repeticiones × "
        f"{N_SPLITS} folds = {N_REPEATS * N_SPLITS} evaluaciones"
    )
    print("Backbone:", backbone, "| Configuración:", requested_model,
          "| Dominio objetivo: PAD-UFES-20 (real)")

    ham_df = collect_image_paths(HAM_DIR, "HAM_claras")
    padufes_df = collect_image_paths(PADUFES_DIR, "PADUFES_real")

    summarize_df(ham_df, "HAM / pieles claras")
    summarize_df(padufes_df, "PADUFES / real")

    padufes_splits = make_repeated_splits(padufes_df)
    ham_splits = make_ham_splits(ham_df)
    done = load_done_keys()
    done_a = extract_done_a(done)

    for repeat, fold, padufes_train_idx, padufes_val_idx in padufes_splits:
        if requested_repeat is not None and repeat != requested_repeat:
            continue

        key = (backbone, requested_model, repeat, fold)
        if key in done:
            print(
                f"⏭️ {backbone} {requested_model} rep{repeat} fold{fold} "
                "ya está guardado"
            )
            continue

        seed = BASE_SEED + repeat * 100 + fold
        set_all_seeds(seed)

        print("\n" + "=" * 65)
        print(
            f"{backbone} | {requested_model} | REPETICIÓN {repeat} | "
            f"FOLD {fold} | SEED {seed}"
        )
        print("=" * 65)

        padufes_train = padufes_df.iloc[padufes_train_idx].reset_index(drop=True)
        padufes_val = padufes_df.iloc[padufes_val_idx].reset_index(drop=True)

        ham_train_idx, ham_val_idx = ham_splits[(repeat, fold)]
        ham_train = ham_df.iloc[ham_train_idx].reset_index(drop=True)
        ham_val = ham_df.iloc[ham_val_idx].reset_index(drop=True)

        if requested_model == "A":
            model, model_path = ensure_a_model(
                backbone, ham_train, ham_val, repeat, fold, done_a
            )
            evaluate_and_save(
                model, padufes_val, "A", repeat, fold,
                model_path, seed,
                extra={"n_train": len(ham_train), "evaluation_domain": "PADUFES",
                       "backbone": backbone}
            )

        elif requested_model == "B":
            model_path = REPRO_MODELS_DIR / f"B_{backbone}_cv_rep{repeat}_fold{fold}.keras"

            # Este fold ya sabemos que NO esta en 'done' (se filtro arriba
            # con 'if key in done: continue'), asi que si el archivo
            # existe es un checkpoint parcial de una corrida interrumpida.
            if model_path.exists():
                print(f"⚠️ Checkpoint parcial encontrado para B (no está en el CSV) "
                      f"— se borra y se reentrena: {model_path.name}")
                model_path.unlink()

            print(f"🚀 Entrenando B [{backbone}] con PAD-UFES-20")
            model = MODEL_BUILDERS[backbone](train_base=False)
            model = train_model(
                model, padufes_train, padufes_val, model_path, EPOCHS_BASE
            )

            evaluate_and_save(
                model, padufes_val, "B", repeat, fold,
                model_path, seed,
                extra={"n_train": len(padufes_train), "evaluation_domain": "PADUFES",
                       "backbone": backbone}
            )

        elif requested_model.startswith("C"):
            n_layers = int(requested_model[1:])
            model_a, _ = ensure_a_model(
                backbone, ham_train, ham_val, repeat, fold, done_a
            )

            model_path = (
                REPRO_MODELS_DIR /
                f"{requested_model}_{backbone}_cv_rep{repeat}_fold{fold}.keras"
            )

            model_c = None
            if model_path.exists():
                print(f"⚠️ Checkpoint parcial encontrado para {requested_model} "
                      f"(no está en el CSV) — se borra y se reentrena: {model_path.name}")
                model_path.unlink()

            print(
                f"🚀 Fine-tuning {requested_model} [{backbone}]: "
                f"{n_layers} capas descongeladas"
            )
            model_c = keras.models.clone_model(model_a)
            model_c.set_weights(model_a.get_weights())
            model_c = unfreeze_last_layers(
                model_c, n_layers=n_layers
            )
            model_c = train_model(
                model_c, padufes_train, padufes_val,
                model_path, EPOCHS_FT
            )

            evaluate_and_save(
                model_c, padufes_val, requested_model,
                repeat, fold, model_path, seed,
                extra={
                    "n_train": len(padufes_train),
                    "evaluation_domain": "PADUFES",
                    "unfreeze_layers": n_layers,
                    "backbone": backbone,
                }
            )

            del model_a
            del model_c

        elif requested_model == "D":
            fused_train = pd.concat(
                [ham_train, padufes_train], ignore_index=True
            )

            model_path = REPRO_MODELS_DIR / f"D_{backbone}_cv_rep{repeat}_fold{fold}.keras"

            if model_path.exists():
                print(f"⚠️ Checkpoint parcial encontrado para D (no está en el CSV) "
                      f"— se borra y se reentrena: {model_path.name}")
                model_path.unlink()

            print(f"🚀 Entrenando D [{backbone}]: HAM10000 ∪ PAD-UFES-20")
            model = MODEL_BUILDERS[backbone](train_base=False)
            model = train_model(
                model, fused_train, padufes_val,
                model_path, EPOCHS_BASE
            )

            evaluate_and_save(
                model, padufes_val, "D", repeat, fold,
                model_path, seed,
                extra={"n_train": len(fused_train), "evaluation_domain": "PADUFES",
                       "backbone": backbone}
            )

        elif requested_model == "E":
            factor = max(
                1, len(ham_train) // max(1, len(padufes_train))
            )

            padufes_over = pd.concat(
                [padufes_train] * factor, ignore_index=True
            )

            fused_balanced = pd.concat(
                [ham_train, padufes_over], ignore_index=True
            ).sample(
                frac=1, random_state=seed
            ).reset_index(drop=True)

            model_path = REPRO_MODELS_DIR / f"E_{backbone}_cv_rep{repeat}_fold{fold}.keras"

            if model_path.exists():
                print(f"⚠️ Checkpoint parcial encontrado para E (no está en el CSV) "
                      f"— se borra y se reentrena: {model_path.name}")
                model_path.unlink()

            print(f"🚀 Entrenando E [{backbone}]: fusión balanceada")
            model = MODEL_BUILDERS[backbone](train_base=False)
            model = train_model(
                model, fused_balanced, padufes_val,
                model_path, EPOCHS_BASE
            )

            evaluate_and_save(
                model, padufes_val, "E", repeat, fold,
                model_path, seed,
                extra={
                    "n_train": len(fused_balanced),
                    "evaluation_domain": "PADUFES",
                    "backbone": backbone,
                }
            )

        if "model" in locals():
            del model

        keras.backend.clear_session()
        gc.collect()

    print(f"\n✅ Resultados acumulados: {RESULTS_CSV}")

    if RESULTS_CSV.exists():
        results = pd.read_csv(RESULTS_CSV)
        current = results[
            (results["model"].astype(str) == requested_model)
            & (results["backbone"].astype(str) == backbone)
        ]

        if not current.empty:
            metric_cols = [
                col for col in [
                    "auc", "accuracy", "sensitivity",
                    "specificity", "precision", "f1",
                    "threshold"
                ]
                if col in current.columns
            ]

            summary = current[metric_cols].agg(
                ["mean", "std", "count"]
            ).T

            summary_path = (
                REPRO_RESULTS_DIR /
                f"resumen_{requested_model}_{backbone}.csv"
            )
            summary.to_csv(summary_path)

            print("\nResumen:")
            print(summary.round(4))
            print(f"✅ Resumen guardado: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repetibilidad por configuración, dominio objetivo = PAD-UFES-20 real."
    )

    parser.add_argument(
        "--backbone",
        default="mobilenetv2",
        choices=list(MODEL_BUILDERS.keys()),
        help="mobilenetv2 (default) o efficientnetb0",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="A, B, C10, C20, C40, C60, C80, D o E",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        choices=range(1, N_REPEATS + 1),
        default=None,
        help="Ejecuta solo una repetición específica.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_configuration(
        backbone=args.backbone,
        requested_model=args.model,
        requested_repeat=args.repeat,
    )
