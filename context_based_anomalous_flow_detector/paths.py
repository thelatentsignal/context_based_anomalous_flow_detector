from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"

#DATA_DIR = PROJECT_ROOT / "data"
#RAW_DATA_DIR = DATA_DIR / "raw"
#PROCESSED_DATA_DIR = DATA_DIR / "processed"
#
MODELS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"
#
## project root = this file's parent’s parent
#PROJECT_ROOT = Path(__file__).resolve().parents[1]
#
#DATA_DIR = PROJECT_ROOT / "data"
#RAW_DIR = DATA_DIR / "raw"
#INTERIM_DIR = DATA_DIR / "interim"
#PROCESSED_DIR = DATA_DIR / "processed"
#
#MODELS_DIR = PROJECT_ROOT / "models"
#REPORTS_DIR = PROJECT_ROOT / "reports"
#FIGURES_DIR = REPORTS_DIR / "figures"
#
## specific files
#RAW_NF_UNSW = RAW_DIR / "f7546561558c07c5_NFV3DATA-A11964_A11964" / "data" / "NF-UNSW-NB15-v3.csv"
#UNIFLOWS_V1 = PROCESSED_DIR / "uniflows_v1.csv"
#