import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent  # only valid for an editable install


def _dir_from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


# Override with CBAFD_DATA_DIR / CBAFD_MODELS_DIR to store data and models outside the repo
# (e.g. on a mounted MinIO bucket). Defaults to <repo>/data and <repo>/models.
BASE_DATA_PATH = _dir_from_env("CBAFD_DATA_DIR", PROJECT_ROOT / "data")
TRAINED_MODELS_DIR = _dir_from_env("CBAFD_MODELS_DIR", PROJECT_ROOT / "models")
