from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

TRAINED_MODELS_DIR = PROJECT_ROOT / "models"
BASE_DATA_PATH = PROJECT_ROOT / "data"
