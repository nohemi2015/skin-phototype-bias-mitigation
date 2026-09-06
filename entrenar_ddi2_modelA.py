"""
entrenar_ddi2_modelA.py

Modelo A: entrenamiento DIRECTO sobre DDI2 con EfficientNetB0.
DDI2:
    C:\5. DDI2 dataset\
        BENIGNAS\
        MALIGNAS\

Protocolo:
- 70% entrenamiento
- 15% validación
- 15% prueba
- EfficientNetB0 con pesos ImageNet
- Binary classification: 0 = BENIGNAS, 1 = MALIGNAS
- Guarda el mejor modelo según val_auc
- Guarda predicciones de test
- Calcula ROC-AUC y Youden
- Guarda curva ROC
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

# ============================================================
# CONFIGURACIÓN
# ============================================================

DATA_DIR = Path(r"C:\5. DDI2 dataset")

PROJECT_DIR = Path(r"C:\CASEIB2026_proyecto")

MODEL_DIR = PROJECT_DIR / "modelos" / "modelo_A_ddi2"
RESULTS_DIR = PROJECT_DIR / "resultados" / "modelo_A_ddi2"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "A_efficientnetb0_ddi2.keras"
PREDICTIONS_CSV = RESULTS_DIR / "predicciones_test_A_efficientnetb0_ddi2.csv"
METRICS_CSV = RESULTS_DIR / "metricas_A_efficientnetb0_ddi2.csv"
ROC_CSV = RESULTS_DIR / "roc_A_efficientnetb0_ddi2.csv"
ROC_PNG = RESULTS_DIR / "ROC_A_efficientnetb0_ddi2.png"
HISTORY_CSV = RESULTS_DIR / "historial_A_efficientnetb0_ddi2.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
SEED = 2026

# ============================================================
# SEMILLAS
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ============================================================
# GPU
# ============================================================

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# ============================================================
# RECOLECTAR IMÁGENES
# ============================================================

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images():
    class_info = {
        "BENIGNAS": 0,
        "MALIGNAS": 1,
    }

    rows = []

    for class_name, label in class_info.items():
        class_dir = DATA_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"No existe la carpeta: {class_dir}"
            )

        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                rows.append({
                    "path": str(path),
                    "label": label,
                    "class_name": class_name,
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            f"No se encontraron imágenes en {DATA_DIR}"
        )

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ============================================================
# PARTICIÓN 70 / 15 / 15
# ============================================================

def make_splits(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=SEED,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df

# ============================================================
# DATASET
# ============================================================

def decode_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.float32)

def make_dataset(df, training=False):
    paths = df["path"].values
    labels = df["label"].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.10),
        ])

        ds = ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ============================================================
# MODELO A - EFFICIENTNETB0
# ============================================================

def build_model():
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        pooling="avg",
    )

    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")

    # EfficientNetB0 de Keras incluye su propio rescaling.
    x = base(inputs, training=False)
    x = keras.layers.Dropout(0.30)(x)
    outputs = keras.layers.Dense(
        1,
        activation="sigmoid",
        name="malignant_probability",
    )(x)

    model = keras.Model(inputs, outputs, name="A_EfficientNetB0_DDI2")

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model

# ============================================================
# EVALUACIÓN
# ============================================================

def evaluate_model(model, test_ds, test_df):
    y_true = test_df["label"].to_numpy().astype(int)
    y_prob = model.predict(test_ds, verbose=1).ravel()

    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr

    youden = tpr + specificity - 1.0
    idx = int(np.argmax(youden))

    optimal_threshold = float(thresholds[idx])
    sensitivity_youden = float(tpr[idx])
    specificity_youden = float(specificity[idx])
    youden_max = float(youden[idx])

    # Métricas al umbral 0.5
    y_pred_05 = (y_prob >= 0.5).astype(int)

    cm_05 = confusion_matrix(
        y_true,
        y_pred_05,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm_05.ravel()

    accuracy = accuracy_score(y_true, y_pred_05)
    precision = precision_score(
        y_true,
        y_pred_05,
        zero_division=0,
    )
    sensitivity = recall_score(
        y_true,
        y_pred_05,
        zero_division=0,
    )
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )
    f1 = f1_score(
        y_true,
        y_pred_05,
        zero_division=0,
    )
    auc = roc_auc_score(y_true, y_prob)

    # Métricas al umbral óptimo de Youden
    y_pred_youden = (y_prob >= optimal_threshold).astype(int)

    cm_youden = confusion_matrix(
        y_true,
        y_pred_youden,
        labels=[0, 1],
    )

    tn_y, fp_y, fn_y, tp_y = cm_youden.ravel()

    accuracy_y = accuracy_score(y_true, y_pred_youden)
    precision_y = precision_score(
        y_true,
        y_pred_youden,
        zero_division=0,
    )
    sensitivity_y = recall_score(
        y_true,
        y_pred_youden,
        zero_division=0,
    )
    specificity_y = (
        tn_y / (tn_y + fp_y)
        if (tn_y + fp_y) > 0
        else 0.0
    )
    f1_y = f1_score(
        y_true,
        y_pred_youden,
        zero_division=0,
    )

    # Guardar predicciones
    pred_df = test_df.copy()
    pred_df["probability_malignant"] = y_prob
    pred_df["prediction_0.5"] = y_pred_05
    pred_df["prediction_youden"] = y_pred_youden

    pred_df.to_csv(
        PREDICTIONS_CSV,
        index=False,
    )

    # Guardar puntos ROC
    roc_df = pd.DataFrame({
        "threshold": thresholds,
        "fpr": fpr,
        "tpr": tpr,
        "specificity": specificity,
        "youden": youden,
    })

    roc_df.to_csv(
        ROC_CSV,
        index=False,
    )

    # Guardar métricas
    metrics = {
        "model": "A",
        "architecture": "EfficientNetB0",
        "dataset": "DDI2",
        "n_test": len(test_df),
        "auc": auc,
        "accuracy_threshold_0.5": accuracy,
        "precision_threshold_0.5": precision,
        "sensitivity_threshold_0.5": sensitivity,
        "specificity_threshold_0.5": specificity,
        "f1_threshold_0.5": f1,
        "threshold_youden": optimal_threshold,
        "accuracy_youden": accuracy_y,
        "precision_youden": precision_y,
        "sensitivity_youden": sensitivity_y,
        "specificity_youden": specificity_y,
        "f1_youden": f1_y,
        "youden_index": youden_max,
        "tn_threshold_0.5": int(tn),
        "fp_threshold_0.5": int(fp),
        "fn_threshold_0.5": int(fn),
        "tp_threshold_0.5": int(tp),
        "tn_youden": int(tn_y),
        "fp_youden": int(fp_y),
        "fn_youden": int(fn_y),
        "tp_youden": int(tp_y),
    }

    pd.DataFrame([metrics]).to_csv(
        METRICS_CSV,
        index=False,
    )

    # ========================================================
    # ROC
    # ========================================================

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"EfficientNetB0 (AUC = {auc:.4f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Model A DDI2")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        ROC_PNG,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\n" + "=" * 65)
    print("RESULTADOS - MODEL A / DDI2")
    print("=" * 65)
    print(f"AUC:                 {auc:.4f}")
    print(f"Accuracy (0.5):      {accuracy:.4f}")
    print(f"Sensitivity (0.5):   {sensitivity:.4f}")
    print(f"Specificity (0.5):   {specificity:.4f}")
    print(f"Precision (0.5):     {precision:.4f}")
    print(f"F1-score (0.5):      {f1:.4f}")
    print("-" * 65)
    print(f"Optimal threshold:   {optimal_threshold:.4f}")
    print(f"Sensitivity (J):     {sensitivity_y:.4f}")
    print(f"Specificity (J):     {specificity_y:.4f}")
    print(f"Youden index:        {youden_max:.4f}")
    print(f"Accuracy (J):        {accuracy_y:.4f}")
    print(f"Precision (J):       {precision_y:.4f}")
    print(f"F1-score (J):        {f1_y:.4f}")
    print("=" * 65)

    return metrics

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 65)
    print("MODEL A - EFFICIENTNETB0 - DDI2")
    print("=" * 65)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"No existe DATA_DIR: {DATA_DIR}"
        )

    df = collect_images()

    print("\nDistribución total:")
    print(df["class_name"].value_counts())

    train_df, val_df, test_df = make_splits(df)

    print("\nParticiones:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test:       {len(test_df)}")

    print("\nTrain:")
    print(train_df["class_name"].value_counts())

    print("\nValidation:")
    print(val_df["class_name"].value_counts())

    print("\nTest:")
    print(test_df["class_name"].value_counts())

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    model = build_model()

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=6,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=str(HISTORY_CSV),
            append=False,
        ),
    ]

    print("\n🚀 Iniciando entrenamiento...")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n🔁 Cargando el mejor modelo...")
    model = keras.models.load_model(MODEL_PATH)

    evaluate_model(
        model,
        test_ds,
        test_df,
    )

    print("\nArchivos generados:")
    print(f"Modelo:       {MODEL_PATH}")
    print(f"Métricas:     {METRICS_CSV}")
    print(f"Predicciones: {PREDICTIONS_CSV}")
    print(f"ROC CSV:      {ROC_CSV}")
    print(f"ROC gráfica:  {ROC_PNG}")
    print(f"Historial:    {HISTORY_CSV}")

    print("\n✅ Entrenamiento de Model A DDI2 terminado.")


if __name__ == "__main__":
    main()
