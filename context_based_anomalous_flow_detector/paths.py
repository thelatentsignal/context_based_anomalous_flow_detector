from pathlib import Path

# project root = this file's parent’s parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# specific files
RAW_NF_UNSW = RAW_DIR / "f7546561558c07c5_NFV3DATA-A11964_A11964" / "data" / "NF-UNSW-NB15-v3_raw.csv"
UNIFLOWS_V1 = PROCESSED_DIR / "uniflows_v1.csv"
# Everything for this data should go into a folder with this name

UNIFLOWS_V1 = PROCESSED_DIR / "uniflows_v1.csv"
INPUT_TO_EMBEDDING_LAYER = INTERIM_DIR / "input_to_embedding_layer.csv"
