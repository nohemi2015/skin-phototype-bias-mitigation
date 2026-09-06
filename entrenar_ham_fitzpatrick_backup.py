"""
entrenar_ham_fitzpatrick.py

Igual que el pipeline original (HAM10000 -> Brown Skin), pero
reemplazando Brown Skin (sintetico) por Fitzpatrick17k (real) como dominio
objetivo. Preentrena/evalua A sobre HAM10000, y adapta hacia Fitzpatrick17k en
las configuraciones B, C10-C80, D, E.

Ejemplos:
    python entrenar_ham_fitzpatrick.py --model C40
    python entrenar_ham_fitzpatrick.py --model A --repeat 1

Modelos admitidos:
    A, B, C10, C20, C40, C60, C80, D, E

Nota metodologica: a diferencia de Brown Skin, Fitzpatrick17k es un dataset real
independiente de HAM10000 (no un derivado por transferencia de estilo),
asi que aqui NO aplica el control de leakage por base_lesion_id que se
necesito para Brown Skin -- la particion aleatoria estratificada de Fitzpatrick17k
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

from config import HAM_DIR, FITZ_DIR, MODELS_DIR, RESULTS_DIR, EPOCHS_BASE, EPOCHS_FT
from dataset import collect_image_paths, summarize_df, make_dataset
from models import build_mobilenetv2, unfreeze_last_layers
from evaluation import predict_scores, evaluate_predictions

N_SPLITS = 5
N_REPEATS = 3
BASE_SEED = 2026

REPRO_MODELS_DIR = MODELS_DIR / "repetibilidad_fitzpatrick"
REPRO_RESULTS_DIR = RESULTS_DIR / "repetibilidad_fitzpatrick"
RESULTS_CSV = REPRO_RESULTS_DIR / "resultados_repetibilidad_fitzpatrick.csv"

REPRO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPRO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    required = {"model", "repeat", "fold"}
    if df.empty or not required.issubset(df.columns):
        return set()

    return {
        (str(row["model"]), int(row["repeat"]), int(row["fold"]))
        for _, row in df.iterrows()
    }


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
        f"\n✅ Guardado: {metrics['model']} | "
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


def ensure_a_model(ham_train, ham_val, repeat, fold):
    model_path = REPRO_MODELS_DIR / f"A_cv_rep{repeat}_fold{fold}.keras"

    if model_path.exists():
        print(f"🔁 Cargando A existente: {model_path.name}")
        return keras.models.load_model(model_path), model_path

    print(f"🚀 Entrenando A | repetición {repeat} | fold {fold}")
    model = build_mobilenetv2(train_base=False)
    model = train_model(
        model, ham_train, ham_val, model_path, EPOCHS_BASE
    )
    return model, model_path


def run_configuration(requested_model, requested_repeat=None):
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
    print("Configuración:", requested_model, "| Dominio objetivo: Fitzpatrick (real)")

    ham_df = collect_image_paths(HAM_DIR, "HAM_claras")
    fitz_df = collect_image_paths(FITZ_DIR, "Fitzpatrick_real")

    summarize_df(ham_df, "HAM / pieles claras")
    summarize_df(fitz_df, "Fitzpatrick / real")

    fitz_splits = make_repeated_splits(fitz_df)
    ham_splits = make_ham_splits(ham_df)
    done = load_done_keys()

    for repeat, fold, fitz_train_idx, fitz_val_idx in fitz_splits:
        if requested_repeat is not None and repeat != requested_repeat:
            continue

        key = (requested_model, repeat, fold)
        if key in done:
            print(
                f"⏭️ {requested_model} rep{repeat} fold{fold} "
                "ya está guardado"
            )
            continue

        seed = BASE_SEED + repeat * 100 + fold
        set_all_seeds(seed)

        print("\n" + "=" * 65)
        print(
            f"{requested_model} | REPETICIÓN {repeat} | "
            f"FOLD {fold} | SEED {seed}"
        )
        print("=" * 65)

        fitz_train = fitz_df.iloc[fitz_train_idx].reset_index(drop=True)
        fitz_val = fitz_df.iloc[fitz_val_idx].reset_index(drop=True)

        ham_train_idx, ham_val_idx = ham_splits[(repeat, fold)]
        ham_train = ham_df.iloc[ham_train_idx].reset_index(drop=True)
        ham_val = ham_df.iloc[ham_val_idx].reset_index(drop=True)

        if requested_model == "A":
            model, model_path = ensure_a_model(
                ham_train, ham_val, repeat, fold
            )
            evaluate_and_save(
                model, fitz_val, "A", repeat, fold,
                model_path, seed,
                extra={"n_train": len(ham_train), "evaluation_domain": "Fitzpatrick"}
            )

        elif requested_model == "B":
            model_path = REPRO_MODELS_DIR / f"B_cv_rep{repeat}_fold{fold}.keras"

            if model_path.exists():
                print(f"🔁 Cargando B existente: {model_path.name}")
                model = keras.models.load_model(model_path)
            else:
                print("🚀 Entrenando B con Fitzpatrick17k")
                model = build_mobilenetv2(train_base=False)
                model = train_model(
                    model, fitz_train, fitz_val, model_path, EPOCHS_BASE
                )

            evaluate_and_save(
                model, fitz_val, "B", repeat, fold,
                model_path, seed,
                extra={"n_train": len(fitz_train), "evaluation_domain": "Fitzpatrick"}
            )

        elif requested_model.startswith("C"):
            n_layers = int(requested_model[1:])
            model_a, _ = ensure_a_model(
                ham_train, ham_val, repeat, fold
            )

            model_path = (
                REPRO_MODELS_DIR /
                f"{requested_model}_cv_rep{repeat}_fold{fold}.keras"
            )

            if model_path.exists():
                print(
                    f"🔁 Cargando {requested_model}: "
                    f"{model_path.name}"
                )
                model_c = keras.models.load_model(model_path)
            else:
                print(
                    f"🚀 Fine-tuning {requested_model}: "
                    f"{n_layers} capas descongeladas"
                )
                model_c = keras.models.clone_model(model_a)
                model_c.set_weights(model_a.get_weights())
                model_c = unfreeze_last_layers(
                    model_c, n_layers=n_layers
                )
                model_c = train_model(
                    model_c, fitz_train, fitz_val,
                    model_path, EPOCHS_FT
                )

            evaluate_and_save(
                model_c, fitz_val, requested_model,
                repeat, fold, model_path, seed,
                extra={
                    "n_train": len(fitz_train),
                    "evaluation_domain": "Fitzpatrick",
                    "unfreeze_layers": n_layers,
                }
            )

            del model_a
            del model_c

        elif requested_model == "D":
            fused_train = pd.concat(
                [ham_train, fitz_train], ignore_index=True
            )

            model_path = REPRO_MODELS_DIR / f"D_cv_rep{repeat}_fold{fold}.keras"

            if model_path.exists():
                print(f"🔁 Cargando D existente: {model_path.name}")
                model = keras.models.load_model(model_path)
            else:
                print("🚀 Entrenando D: HAM10000 ∪ Fitzpatrick17k")
                model = build_mobilenetv2(train_base=False)
                model = train_model(
                    model, fused_train, fitz_val,
                    model_path, EPOCHS_BASE
                )

            evaluate_and_save(
                model, fitz_val, "D", repeat, fold,
                model_path, seed,
                extra={"n_train": len(fused_train), "evaluation_domain": "Fitzpatrick"}
            )

        elif requested_model == "E":
            factor = max(
                1, len(ham_train) // max(1, len(fitz_train))
            )

            fitz_over = pd.concat(
                [fitz_train] * factor, ignore_index=True
            )

            fused_balanced = pd.concat(
                [ham_train, fitz_over], ignore_index=True
            ).sample(
                frac=1, random_state=seed
            ).reset_index(drop=True)

            model_path = REPRO_MODELS_DIR / f"E_cv_rep{repeat}_fold{fold}.keras"

            if model_path.exists():
                print(f"🔁 Cargando E existente: {model_path.name}")
                model = keras.models.load_model(model_path)
            else:
                print("🚀 Entrenando E: fusión balanceada")
                model = build_mobilenetv2(train_base=False)
                model = train_model(
                    model, fused_balanced, fitz_val,
                    model_path, EPOCHS_BASE
                )

            evaluate_and_save(
                model, fitz_val, "E", repeat, fold,
                model_path, seed,
                extra={
                    "n_train": len(fused_balanced),
                    "evaluation_domain": "Fitzpatrick"
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
            results["model"].astype(str) == requested_model
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
                f"resumen_{requested_model}.csv"
            )
            summary.to_csv(summary_path)

            print("\nResumen:")
            print(summary.round(4))
            print(f"✅ Resumen guardado: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repetibilidad por configuración, dominio objetivo = Fitzpatrick real."
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
        requested_model=args.model,
        requested_repeat=args.repeat,
    )
