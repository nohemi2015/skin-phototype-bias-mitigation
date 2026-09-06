from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "datos_piel"
HAM_DIR = DATA_DIR / "pieles_claras"
BS_DIR = DATA_DIR / "pieles_oscuras"
DDI_DIR = DATA_DIR / "DDI_clasificado"
FITZ_DIR = DATA_DIR / "Fitzpatrick_clasificado"
PADUFES_DIR = DATA_DIR / "PADUFES_clasificado"

MODELS_DIR = PROJECT_DIR / "modelos"
RESULTS_DIR = PROJECT_DIR / "resultados"
PREDICTIONS_DIR = RESULTS_DIR / "predicciones"
FIGURES_DIR = RESULTS_DIR / "figuras"
LOGS_DIR = PROJECT_DIR / "logs"
PARTITIONS_DIR = PROJECT_DIR / "particiones"

for d in [
    MODELS_DIR,
    RESULTS_DIR,
    PREDICTIONS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    PARTITIONS_DIR,
]:
    d.mkdir(exist_ok=True, parents=True)

CLASS_NAMES = {"BENIGNAS": 0, "MALIGNAS": 1}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

EPOCHS_BASE = 20
EPOCHS_FT = 20
LR_BASE = 1e-4
LR_FT = 1e-5

RUN_MODELS = ["C"]

# C10, C20 y C40 serán omitidos automáticamente por is_done().
# Solo se entrenarán C60 y C80.
UNFREEZE_SWEEP = [80]

RESULTS_CSV = RESULTS_DIR / "resultados_incrementales.csv"
SUMMARY_CSV = RESULTS_DIR / "tabla_resumen_metricas.csv"
