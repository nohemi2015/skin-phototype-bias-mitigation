import tensorflow as tf
from tensorflow import keras
from config import IMG_SIZE, LR_BASE, LR_FT


def build_mobilenetv2(train_base=False, lr=LR_BASE):
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = train_base

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def build_efficientnetb0(train_base=False, lr=LR_BASE):
    """
    OJO con el preprocesamiento: a diferencia de MobileNetV2, el
    EfficientNetB0 de Keras trae su propia normalizacion interna
    (equivalente a ImageNet mean/std) y espera pixeles en rango
    [0, 255], NO en [0, 1]. Como dataset.py ya normaliza todo a [0, 1]
    (compartido con MobileNetV2), aqui se agrega una capa Rescaling(255)
    para deshacer esa normalizacion antes de entrar al backbone. Sin
    esto, EfficientNetB0 rinde mucho peor porque recibe la imagen ya
    des-escalada dos veces.
    """
    base = keras.applications.EfficientNetB0(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = train_base

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = keras.layers.Rescaling(255.0)(inputs)  # deshace el /255.0 de dataset.py
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def unfreeze_last_layers(model, n_layers=20, lr=LR_FT):
    # Busca el backbone interno (MobileNetV2 o EfficientNetB0, ambos son
    # keras.Model anidados dentro del modelo completo).
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base = layer
            base.trainable = True
            total = len(base.layers)
            for i, sub in enumerate(base.layers):
                sub.trainable = (i >= total - n_layers)
            break

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model
