"""
entrenar_leakage_safe_v2.py  — Script autocontenido
====================================================
v2 (21-oct-2026), cambios respecto a entrenar_leakage_safe.py:
  - EfficientNetB0: se agrega Rescaling(255) (el backbone de Keras espera
    pixeles en [0,255]; sin esto la imagen se normaliza dos veces).
  - A y B respetan --arch (antes A y B siempre eran MobileNetV2).
  - load_done(): filas sin 'arch' cuentan como mobilenetv2 (antes se
    re-entrenaban; asi se duplicaron los folds de D).
  - --train_final: guarda el umbral de Youden del conjunto de VALIDACION
    (umbral congelado para el zero-shot), evalua el control de Brown Skin
    SOLO con imagenes BS del hold-out, y guarda columna 'source'.
  Los resultados ya existentes de MobileNetV2 no cambian ni se re-entrenan.

VERSION CORREGIDA segun reporte Elicit (sep-2026).
No requiere config.py, models.py, evaluation.py ni dataset.py.
Todo esta integrado en este unico archivo.

Correcciones implementadas:
  1. Split leakage-safe unificado HAM+BS a nivel de lesion_id
  2. Guardado de predicciones crudas por fold (para ROC empiricas)
  3. Modelo final sin cherry-picking (--train_final)
  4. AUC group-aware como estimado primario

Uso:
    python entrenar_leakage_safe.py --model C40
    python entrenar_leakage_safe.py --model C40 --repeat 1
    python entrenar_leakage_safe.py --model C40 --train_final
"""

import os, gc, re, random, argparse, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score

# ── Configuracion (equivalente a config.py) ───────────────────────────────────
PROJECT_DIR  = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_DIR / "datos_piel"
HAM_DIR      = DATA_DIR / "pieles_claras"
BS_DIR       = DATA_DIR / "pieles_oscuras"

MODELS_DIR   = PROJECT_DIR / "modelos"   / "leakage_safe"
RESULTS_DIR  = PROJECT_DIR / "resultados" / "leakage_safe"
PRED_DIR     = RESULTS_DIR / "predicciones"
LEDGER_PATH  = RESULTS_DIR / "split_ledger.csv"
RESULTS_CSV  = RESULTS_DIR / "resultados_leakage_safe.csv"

for d in [MODELS_DIR, RESULTS_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HAM_METADATA_PATH = PROJECT_DIR / "HAM10000_metadata.csv"

CLASS_NAMES  = {"BENIGNAS": 0, "MALIGNAS": 1}
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS_BASE  = 20
EPOCHS_FT    = 20
LR_BASE      = 1e-4
LR_FT        = 1e-5
N_SPLITS     = 5
N_REPEATS    = 3
BASE_SEED    = 2026


# ── Seeds ─────────────────────────────────────────────────────────────────────
def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ── Carga de imagenes ─────────────────────────────────────────────────────────
def collect_image_paths(root_dir, domain_name):
    root_dir = Path(root_dir)
    rows = []
    for class_name, label in CLASS_NAMES.items():
        class_dir = root_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"No existe: {class_dir}")
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append({"filepath": str(path.resolve()),
                              "label": int(label),
                              "class_name": class_name,
                              "domain": domain_name})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No se encontraron imagenes en {root_dir}")
    return df.sample(frac=1, random_state=BASE_SEED).reset_index(drop=True)


# ── Provenance map HAM10000 ───────────────────────────────────────────────────
IMAGE_ID_RE = re.compile(r"ISIC[_]?(\d+)", re.IGNORECASE)

def build_provenance_map(meta_path):
    meta = pd.read_csv(meta_path)
    meta["image_id"] = meta["image_id"].str.strip().str.upper()
    prov = dict(zip(meta["image_id"], meta["lesion_id"]))
    print(f"Provenance map: {len(prov)} image_ids cargados")
    return prov

def extract_image_id(filepath):
    m = IMAGE_ID_RE.search(os.path.basename(filepath))
    return f"ISIC_{m.group(1)}" if m else None

def add_lesion_id(df, prov, domain):
    df = df.copy()
    def get_lid(fp):
        iid = extract_image_id(fp)
        if iid and iid in prov:
            return prov[iid]
        return iid if iid else f"__{domain}__{os.path.basename(fp)}"
    df["lesion_id"] = df["filepath"].apply(get_lid)
    n = df["lesion_id"].str.startswith("HAM_").sum()
    print(f"{domain}: {n}/{len(df)} mapeadas a lesion_id | "
          f"{df['lesion_id'].nunique()} lesiones unicas")
    return df


# ── Split unificado leakage-safe ──────────────────────────────────────────────
def make_unified_splits(ham_df, bs_df):
    """
    Construye splits sobre el grafo combinado HAM+BS agrupando por lesion_id.
    Garantia: una lesion y TODAS sus derivadas sinteticas van siempre
    al mismo fold — en ambos datasets simultaneamente.
    """
    h = ham_df[["lesion_id","label"]].copy(); h["source"]="ham"
    h["orig_idx"] = np.arange(len(ham_df))
    b = bs_df[["lesion_id","label"]].copy();  b["source"]="bs"
    b["orig_idx"] = np.arange(len(bs_df))
    combined = pd.concat([h, b], ignore_index=True)

    groups = combined["lesion_id"].values
    labels = combined["label"].values
    print(f"\nSplit unificado: {combined['lesion_id'].nunique()} lesiones unicas "
          f"| HAM {len(ham_df)} imgs | BS {len(bs_df)} imgs")

    all_splits = []
    for repeat in range(1, N_REPEATS+1):
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                     random_state=BASE_SEED+repeat)
        for fold, (tr_idx, va_idx) in enumerate(
                sgkf.split(combined, labels, groups=groups), start=1):
            # Verificacion de seguridad
            overlap = set(groups[tr_idx]) & set(groups[va_idx])
            if overlap:
                raise RuntimeError(f"FUGA rep={repeat} fold={fold}: {list(overlap)[:5]}")
            tr_rows = combined.iloc[tr_idx]
            va_rows = combined.iloc[va_idx]
            all_splits.append((
                repeat, fold,
                tr_rows[tr_rows["source"]=="ham"]["orig_idx"].values,
                va_rows[va_rows["source"]=="ham"]["orig_idx"].values,
                tr_rows[tr_rows["source"]=="bs"]["orig_idx"].values,
                va_rows[va_rows["source"]=="bs"]["orig_idx"].values,
            ))
    print(f"Splits generados: {len(all_splits)} "
          f"({N_SPLITS} folds x {N_REPEATS} repeats)")
    return all_splits

def save_ledger(ham_df, bs_df, splits):
    if LEDGER_PATH.exists():
        print(f"Ledger ya existe: {LEDGER_PATH}"); return
    rows = []
    for rep, fold, h_tr, h_va, b_tr, b_va in splits:
        for part, idx, src, df in [("train",h_tr,"ham",ham_df),
                                    ("val",  h_va,"ham",ham_df),
                                    ("train",b_tr,"bs", bs_df),
                                    ("val",  b_va,"bs", bs_df)]:
            for i in idx:
                rows.append({"source":src,"repeat":rep,"fold":fold,
                              "partition":part,
                              "filepath":df.iloc[i]["filepath"],
                              "lesion_id":df.iloc[i]["lesion_id"],
                              "label":df.iloc[i]["label"]})
    pd.DataFrame(rows).to_csv(LEDGER_PATH, index=False)
    print(f"Ledger guardado: {LEDGER_PATH} ({len(rows)} filas)")


# ── Dataset TF ────────────────────────────────────────────────────────────────
def decode_image(fp, label, augment=False):
    img = tf.io.read_file(fp)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.15)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    return img, tf.cast(label, tf.float32)

def make_dataset(df, augment=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (df["filepath"].values, df["label"].values.astype("float32")))
    if shuffle:
        ds = ds.shuffle(len(df), seed=BASE_SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda x,y: decode_image(x,y,augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ── Modelos ───────────────────────────────────────────────────────────────────
def build_mobilenetv2(train_base=False, lr=LR_BASE):
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE,3), include_top=False, weights="imagenet")
    base.trainable = train_base
    inp = keras.Input(shape=(*IMG_SIZE,3))
    x = base(inp, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model

def build_efficientnetb0(train_base=False, lr=LR_BASE):
    base = keras.applications.EfficientNetB0(
        input_shape=(*IMG_SIZE,3), include_top=False, weights="imagenet")
    base.trainable = train_base
    inp = keras.Input(shape=(*IMG_SIZE,3))
    x = keras.layers.Rescaling(255.0)(inp)   # v2: deshace el /255 de decode_image
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model

def build_model(arch, train_base=False, lr=LR_BASE):
    """Factory: retorna el modelo según la arquitectura especificada."""
    if arch == "efficientnetb0":
        return build_efficientnetb0(train_base=train_base, lr=lr)
    return build_mobilenetv2(train_base=train_base, lr=lr)

def unfreeze_last_layers(model, n_layers, lr=LR_FT):
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            layer.trainable = True
            total = len(layer.layers)
            for i, sub in enumerate(layer.layers):
                sub.trainable = (i >= total - n_layers)
            break
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model


# ── Evaluacion ────────────────────────────────────────────────────────────────
def predict_scores(model, ds):
    y_true, y_score = [], []
    for x, y in ds:
        y_true.extend(np.asarray(y).reshape(-1))
        y_score.extend(model.predict(x, verbose=0).reshape(-1))
    return np.array(y_true, dtype=int), np.array(y_score, dtype=float)

def youden_threshold(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return float(thr[idx])

def compute_metrics(y_true, y_score):
    auc  = float(roc_auc_score(y_true, y_score))
    thr  = youden_threshold(y_true, y_score)
    pred = (y_score >= thr).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0.
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.
    prec = float(precision_score(y_true, pred, zero_division=0))
    f1   = float(f1_score(y_true, pred, zero_division=0))
    acc  = float(accuracy_score(y_true, pred))
    return {"auc":auc,"accuracy":acc,"sensitivity":float(sens),
            "specificity":float(spec),"precision":prec,"f1":f1,
            "threshold":thr,"youden":float(sens+spec-1),
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}


# ── I/O de resultados ─────────────────────────────────────────────────────────
def load_done():
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size==0:
        return set()
    df = pd.read_csv(RESULTS_CSV)
    if df.empty or not {"model","repeat","fold"}.issubset(df.columns):
        return set()
    if "arch" in df.columns:
        arch_col = df["arch"].fillna("mobilenetv2").astype(str)
        return {(str(r["model"]),int(r["repeat"]),int(r["fold"]),a)
                for (_,r),a in zip(df.iterrows(), arch_col)}
    return {(str(r["model"]),int(r["repeat"]),int(r["fold"]),"mobilenetv2")
            for _,r in df.iterrows()}

def save_result(metrics):
    row = pd.DataFrame([metrics])
    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size>0:
        row = pd.concat([pd.read_csv(RESULTS_CSV), row],
                         ignore_index=True, sort=False)
    row.to_csv(RESULTS_CSV, index=False)
    print(f"✅ {metrics['model']} rep{metrics['repeat']} "
          f"fold{metrics['fold']} AUC={metrics['auc']:.4f}")

def save_predictions(y_true, y_score, model_name, repeat, fold, extra=None,
                     arch="mobilenetv2"):
    """Guarda predicciones crudas para curvas ROC empiricas."""
    tag = "" if arch == "mobilenetv2" else f"_{arch}"   # v2: no pisar MobileNetV2
    path = PRED_DIR / f"pred_{model_name}{tag}_rep{repeat}_fold{fold}.csv"
    df = pd.DataFrame({"y_true":y_true, "y_score":y_score,
                        "model":model_name, "repeat":repeat, "fold":fold})
    if extra:
        for k,v in extra.items(): df[k] = v
    df.to_csv(path, index=False)
    return path


# ── Callbacks ─────────────────────────────────────────────────────────────────
def make_callbacks(model_path):
    hist = RESULTS_DIR / f"hist_{model_path.stem}.csv"
    return [
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=6,
            mode="max", restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_auc",
            mode="max", save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger(str(hist), append=False),
    ]


# ── Entrenamiento ─────────────────────────────────────────────────────────────
def train_model(model, train_df, val_df, model_path, epochs):
    tr = make_dataset(train_df, augment=True,  shuffle=True)
    va = make_dataset(val_df,   augment=False, shuffle=False)
    model.fit(tr, validation_data=va, epochs=epochs,
              callbacks=make_callbacks(model_path), verbose=2)
    return model

def ensure_a(ham_train, ham_val, repeat, fold, arch="mobilenetv2"):
    mp = MODELS_DIR / f"A_{arch}_rep{repeat}_fold{fold}.keras"
    if mp.exists():
        print(f"🔁 Cargando A: {mp.name}")
        return keras.models.load_model(mp), mp
    print(f"🚀 Entrenando A ({arch}, backbone congelado)")
    m = build_model(arch, train_base=False)
    m = train_model(m, ham_train, ham_val, mp, EPOCHS_BASE)
    return m, mp

def eval_and_save(model, val_df, model_name, repeat, fold, model_path,
                   seed, arch="mobilenetv2", extra=None):
    ds = make_dataset(val_df, augment=False, shuffle=False)
    y_true, y_score = predict_scores(model, ds)
    metrics = compute_metrics(y_true, y_score)
    pred_path = save_predictions(y_true, y_score, model_name, repeat, fold, arch=arch)
    metrics.update({"model":model_name,"repeat":repeat,"fold":fold,
                     "seed":seed,"model_file":str(model_path),
                     "pred_file":str(pred_path),"n_val":len(val_df),
                     "split_type":"leakage_safe_unified",
                     "arch":arch})
    if extra: metrics.update(extra)
    save_result(metrics)
    return y_true, y_score


# ── Modelo final sin cherry-picking ──────────────────────────────────────────
def train_final(ham_df, bs_df, requested_model, arch="mobilenetv2"):
    print("\n"+"="*65)
    print("MODELO FINAL (sin cherry-picking) — para validacion externa")
    print("="*65)
    h = ham_df.copy(); h["source"]="ham"
    b = bs_df.copy();  b["source"]="bs"
    combined = pd.concat([h, b], ignore_index=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=BASE_SEED)
    dev_idx, test_idx = next(gss.split(combined, groups=combined["lesion_id"]))
    dev_df  = combined.iloc[dev_idx].reset_index(drop=True)
    test_df = combined.iloc[test_idx].reset_index(drop=True)

    overlap = set(dev_df["lesion_id"]) & set(test_df["lesion_id"])
    if overlap:
        raise RuntimeError(f"Fuga en split final: {len(overlap)} lesiones")

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=BASE_SEED+1)
    tr_idx, va_idx = next(gss2.split(dev_df, groups=dev_df["lesion_id"]))
    train_df = dev_df.iloc[tr_idx].reset_index(drop=True)
    val_df   = dev_df.iloc[va_idx].reset_index(drop=True)

    ham_tr = train_df[train_df["source"]=="ham"].reset_index(drop=True)
    ham_va = val_df[val_df["source"]=="ham"].reset_index(drop=True)
    bs_tr  = train_df[train_df["source"]=="bs"].reset_index(drop=True)
    bs_va  = val_df[val_df["source"]=="bs"].reset_index(drop=True)

    print(f"  Dev: {len(dev_df)} imgs | Test: {len(test_df)} imgs")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    set_seeds(BASE_SEED)
    mp = MODELS_DIR / f"FINAL_{requested_model}_{arch}.keras"

    if requested_model == "A":
        m = build_model(arch, train_base=False)
        m = train_model(m, ham_tr, ham_va, mp, EPOCHS_BASE)
    elif requested_model == "B":
        m = build_model(arch, train_base=False)
        m = train_model(m, bs_tr, bs_va, mp, EPOCHS_BASE)
    elif requested_model.startswith("C"):
        n = int(requested_model[1:])
        mp_a = MODELS_DIR / f"FINAL_A_base_{arch}.keras"
        ma = build_model(arch, train_base=False)
        ma = train_model(ma, ham_tr, ham_va, mp_a, EPOCHS_BASE)
        m = keras.models.clone_model(ma); m.set_weights(ma.get_weights())
        m = unfreeze_last_layers(m, n_layers=n)
        m = train_model(m, bs_tr, bs_va, mp, EPOCHS_FT)
        del ma
    elif requested_model == "D":
        fused = pd.concat([ham_tr, bs_tr], ignore_index=True)
        m = build_model(arch, train_base=False)
        m = train_model(m, fused, bs_va, mp, EPOCHS_BASE)
    elif requested_model == "E":
        factor = max(1, len(ham_tr)//max(1, len(bs_tr)))
        bs_over = pd.concat([bs_tr]*factor, ignore_index=True)
        fused = pd.concat([ham_tr, bs_over], ignore_index=True).sample(
            frac=1, random_state=BASE_SEED).reset_index(drop=True)
        m = build_model(arch, train_base=False)
        m = train_model(m, fused, bs_va, mp, EPOCHS_BASE)

    # v2: recargar el mejor checkpoint (el que se usara en el zero-shot)
    m = keras.models.load_model(mp)

    # v2: umbral congelado = Youden sobre la VALIDACION de Brown Skin
    # (nunca sobre el test ni sobre las cohortes externas)
    yv_true, yv_score = predict_scores(m, make_dataset(bs_va, augment=False, shuffle=False))
    frozen_thr = youden_threshold(yv_true, yv_score)

    # Control same-distribution: SOLO imagenes Brown Skin del hold-out
    bs_test = test_df[test_df["source"]=="bs"].reset_index(drop=True)
    y_true, y_score = predict_scores(m, make_dataset(bs_test, augment=False, shuffle=False))
    metrics = compute_metrics(y_true, y_score)              # umbral Youden del propio test
    pred_f = (y_score >= frozen_thr).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_true, pred_f, labels=[0,1]).ravel()
    metrics.update({"frozen_threshold": frozen_thr,
                    "sens_at_frozen": float(tp/(tp+fn)) if (tp+fn) else 0.,
                    "spec_at_frozen": float(tn/(tn+fp)) if (tn+fp) else 0.})
    print(f"\n📊 CONTROL BROWN SKIN (hold-out, solo BS, n={len(bs_test)}):")
    for k,v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v,float) else f"  {k}: {v}")

    pd.DataFrame({"y_true":y_true,"y_score":y_score,
                   "filepath":bs_test["filepath"].values,
                   "lesion_id":bs_test["lesion_id"].values,
                   "source":"bs","model":f"FINAL_{requested_model}",
                   "partition":"test_holdout"}).to_csv(
        PRED_DIR/f"pred_FINAL_{requested_model}_{arch}_test_BS.csv", index=False)

    info = {"model_file": str(mp), "arch": arch, "config": requested_model,
            "frozen_threshold_from_bs_val": frozen_thr,
            "n_train": len(train_df), "n_val": len(val_df),
            "n_test_total": len(test_df), "n_test_bs": len(bs_test),
            "control_bs_auc": metrics["auc"]}
    with open(RESULTS_DIR/f"FINAL_{requested_model}_{arch}.json","w") as f:
        json.dump(info, f, indent=2)
    print(f"Info del modelo final: {RESULTS_DIR/f'FINAL_{requested_model}_{arch}.json'}")

    save_result({**metrics,"model":f"FINAL_{requested_model}",
                  "repeat":0,"fold":0,"n_val":len(bs_test),"arch":arch,
                  "split_type":"leakage_safe_final_holdout"})
    print(f"\n✅ Modelo final: {mp}")
    print("   Usar este modelo para validacion externa zero-shot.")


# ── Pipeline principal ────────────────────────────────────────────────────────
def run(requested_model, requested_repeat=None, do_train_final=False, arch="mobilenetv2"):
    set_seeds(BASE_SEED)
    print("Cargando datos...")
    ham_df = collect_image_paths(HAM_DIR, "HAM_claras")
    bs_df  = collect_image_paths(BS_DIR,  "BS_oscuras")

    prov   = build_provenance_map(HAM_METADATA_PATH)
    ham_df = add_lesion_id(ham_df, prov, "HAM")
    bs_df  = add_lesion_id(bs_df,  prov, "BS")

    if do_train_final:
        train_final(ham_df, bs_df, requested_model, arch=arch); return

    splits = make_unified_splits(ham_df, bs_df)
    save_ledger(ham_df, bs_df, splits)
    done   = load_done()

    for rep, fold, h_tr, h_va, b_tr, b_va in splits:
        if requested_repeat is not None and rep != requested_repeat:
            continue
        key = (requested_model, rep, fold, arch)
        if key in done:
            print(f"⏭️  {requested_model} rep{rep} fold{fold} ya guardado"); continue

        seed = BASE_SEED + rep*100 + fold
        set_seeds(seed)
        print(f"\n{'='*65}\n{requested_model} | REP {rep} | FOLD {fold} | SEED {seed}\n{'='*65}")

        ham_train = ham_df.iloc[h_tr].reset_index(drop=True)
        ham_val   = ham_df.iloc[h_va].reset_index(drop=True)
        bs_train  = bs_df.iloc[b_tr].reset_index(drop=True)
        bs_val    = bs_df.iloc[b_va].reset_index(drop=True)

        if requested_model == "A":
            m, mp = ensure_a(ham_train, ham_val, rep, fold, arch=arch)
            eval_and_save(m, bs_val, "A", rep, fold, mp, seed,
                           extra={"n_train":len(ham_train),"eval_domain":"BS","arch":arch})

        elif requested_model == "B":
            # v2: el nombre de MobileNetV2 no cambia (compatibilidad con lo ya corrido)
            name = f"B_rep{rep}_fold{fold}.keras" if arch=="mobilenetv2" \
                   else f"B_{arch}_rep{rep}_fold{fold}.keras"
            mp = MODELS_DIR/name
            m  = keras.models.load_model(mp) if mp.exists() \
                 else train_model(build_model(arch), bs_train, bs_val, mp, EPOCHS_BASE)
            eval_and_save(m, bs_val, "B", rep, fold, mp, seed,
                           extra={"n_train":len(bs_train),"eval_domain":"BS","arch":arch})

        elif requested_model.startswith("C"):
            n  = int(requested_model[1:])
            ma, _ = ensure_a(ham_train, ham_val, rep, fold, arch=arch)
            mp = MODELS_DIR/f"{requested_model}_{arch}_rep{rep}_fold{fold}.keras"
            if mp.exists():
                mc = keras.models.load_model(mp)
            else:
                mc = keras.models.clone_model(ma); mc.set_weights(ma.get_weights())
                mc = unfreeze_last_layers(mc, n_layers=n)
                mc = train_model(mc, bs_train, bs_val, mp, EPOCHS_FT)
            eval_and_save(mc, bs_val, requested_model, rep, fold, mp, seed,
                           extra={"n_train":len(bs_train),"eval_domain":"BS",
                                   "unfreeze_layers":n,"arch":arch})
            del ma, mc

        elif requested_model == "D":
            fused = pd.concat([ham_train, bs_train], ignore_index=True)
            mp = MODELS_DIR/f"D_{arch}_rep{rep}_fold{fold}.keras"
            m  = keras.models.load_model(mp) if mp.exists() \
                 else train_model(build_model(arch), fused, bs_val, mp, EPOCHS_BASE)
            eval_and_save(m, bs_val, "D", rep, fold, mp, seed,
                           extra={"n_train":len(fused),"eval_domain":"BS","arch":arch})

        elif requested_model == "E":
            factor = max(1, len(ham_train)//max(1,len(bs_train)))
            bs_over = pd.concat([bs_train]*factor, ignore_index=True)
            fused = pd.concat([ham_train, bs_over], ignore_index=True).sample(
                frac=1, random_state=seed).reset_index(drop=True)
            mp = MODELS_DIR/f"E_{arch}_rep{rep}_fold{fold}.keras"
            m  = keras.models.load_model(mp) if mp.exists() \
                 else train_model(build_model(arch), fused, bs_val, mp, EPOCHS_BASE)
            eval_and_save(m, bs_val, "E", rep, fold, mp, seed,
                           extra={"n_train":len(fused),"eval_domain":"BS",
                                   "arch":arch})

        if "m" in dir(): del m
        keras.backend.clear_session(); gc.collect()

    # Resumen final
    if RESULTS_CSV.exists():
        res = pd.read_csv(RESULTS_CSV)
        arch_col = res["arch"].fillna("mobilenetv2") if "arch" in res else "mobilenetv2"
        cur = res[(res["model"].astype(str)==requested_model) & (arch_col==arch)]
        cur = cur.drop_duplicates(["model","repeat","fold"], keep="last")
        if not cur.empty:
            cols = [c for c in ["auc","accuracy","sensitivity",
                                  "specificity","precision","f1"] if c in cur.columns]
            summ = cur[cols].agg(["mean","std","count"]).T
            tag = "" if arch=="mobilenetv2" else f"_{arch}"
            summ.to_csv(RESULTS_DIR/f"resumen_{requested_model}{tag}.csv")
            print(f"\n{'='*65}")
            print(f"RESUMEN {requested_model} — AUC group-aware (ESTIMADO PRIMARIO):")
            print(summ.round(4))
            print("\n⚠️  Este AUC es el estimado primario del paper.")
            print("   El AUC 0.816 previo (image-level) queda como sensibilidad.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                     help="A B C10 C20 C40 C60 C80 D E")
    ap.add_argument("--repeat", type=int, default=None,
                     choices=range(1, N_REPEATS+1))
    ap.add_argument("--train_final", action="store_true",
                     help="Entrena modelo final sin cherry-picking")
    ap.add_argument("--arch", default="mobilenetv2",
                     choices=["mobilenetv2","efficientnetb0"],
                     help="Arquitectura backbone (default: mobilenetv2)")
    args = ap.parse_args()
    run(args.model, args.repeat, args.train_final, args.arch)
