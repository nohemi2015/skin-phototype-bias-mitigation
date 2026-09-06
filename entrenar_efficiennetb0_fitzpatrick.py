import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

DATASET_DIR = r"C:\CASEIB2026_proyecto\Fitzpatrick17k\particiones"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR   = os.path.join(DATASET_DIR, "val")
TEST_DIR  = os.path.join(DATASET_DIR, "test")

OUTPUT_DIR = r"C:\CASEIB2026_proyecto\resultados_efficientnetb0"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

EPOCHS_HEAD = 20
EPOCHS_FINE = 20

# ============================================================
# 2. SEMILLAS PARA REPRODUCIBILIDAD
# ============================================================

tf.keras.utils.set_random_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU disponibles:", tf.config.list_physical_devices("GPU"))

# ============================================================
# 3. CARGAR DATASETS
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClases detectadas:")
print(train_ds.class_names)

# ============================================================
# 4. OPTIMIZACIÓN DEL PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
], name="data_augmentation")

# ============================================================
# 6. MODELO EfficientNetB0
# ============================================================

base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Congelar inicialmente la base
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(
    inputs,
    outputs,
    name="EfficientNetB0_Fitzpatrick17k"
)

model.summary()

# ============================================================
# 7. COMPILACIÓN - FASE 1
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc")
    ]
)

# ============================================================
# 8. CALLBACKS
# ============================================================

best_model_path = os.path.join(
    OUTPUT_DIR,
    "EfficientNetB0_best.keras"
)

callbacks = [

    keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================
# 9. ENTRENAMIENTO - FASE 1
# ============================================================

print("\n==============================================")
print("FASE 1: TRANSFER LEARNING")
print("==============================================")

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks
)

# ============================================================
# 10. FINE-TUNING
# ============================================================

print("\n==============================================")
print("FASE 2: FINE-TUNING")
print("==============================================")

base_model.trainable = True

# Congelar las primeras capas
fine_tune_from = 200

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

# BatchNormalization se mantiene congelada
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc")
    ]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=callbacks
)

# ============================================================
# 11. CARGAR EL MEJOR MODELO
# ============================================================

print("\nCargando el mejor modelo...")

model = keras.models.load_model(
    best_model_path
)

# ============================================================
# 12. EVALUACIÓN EN TEST
# ============================================================

print("\n==============================================")
print("EVALUACIÓN FINAL EN TEST")
print("==============================================")

test_loss, test_accuracy, test_auc = model.evaluate(
    test_ds,
    verbose=1
)

print("\nResultados:")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test AUC:", test_auc)

# ============================================================
# 13. OBTENER PREDICCIONES
# ============================================================

y_true = []
y_prob = []

for images, labels in test_ds:

    probabilities = model.predict(
        images,
        verbose=0
    )

    y_prob.extend(
        probabilities.flatten()
    )

    y_true.extend(
        labels.numpy().flatten()
    )

y_true = np.array(y_true).astype(int)
y_prob = np.array(y_prob)

# Umbral 0.5
y_pred = (y_prob >= 0.5).astype(int)

# ============================================================
# 14. MÉTRICAS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    y_prob
)

# Matriz de confusión
cm = confusion_matrix(
    y_true,
    y_pred
)

TN, FP, FN, TP = cm.ravel()

specificity = TN / (TN + FP)

# ============================================================
# 15. MOSTRAR RESULTADOS
# ============================================================

print("\n==============================================")
print("MÉTRICAS FINALES")
print("==============================================")

print(f"Accuracy:     {accuracy:.4f}")
print(f"Precision:    {precision:.4f}")
print(f"Sensibilidad: {recall:.4f}")
print(f"Especificidad:{specificity:.4f}")
print(f"F1-score:     {f1:.4f}")
print(f"ROC-AUC:      {auc:.4f}")

print("\nMatriz de confusión:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "BENIGNAS",
            "MALIGNAS"
        ],
        zero_division=0
    )
)

# ============================================================
# 16. GUARDAR PREDICCIONES
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_true.npy"
    ),
    y_true
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_prob.npy"
    ),
    y_prob
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_pred.npy"
    ),
    y_pred
)

print("\nPredicciones guardadas.")

# ============================================================
# 17. GUARDAR MODELO FINAL
# ============================================================

final_model_path = os.path.join(
    OUTPUT_DIR,
    "EfficientNetB0_final.keras"
)

model.save(final_model_path)

print("\nModelo final guardado en:")
print(final_model_path)

print("\n==============================================")
print("ENTRENAMIENTO Y EVALUACIÓN TERMINADOS")
print("==============================================")