from pathlib import Path
from typing import Any

import yaml

from context_based_anomalous_flow_detector.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def get_dataset_params(
    dataset_id: str = "unsw_nb15",
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config = load_config(config_path)

    params = config["datasets"][dataset_id].copy()

    for key in ["raw_input", "processed_data_dir"]:
        if key in params:
            params[key] = str(resolve_path(params[key]))

    return params

#from importlib.resources import files
#from pathlib import Path
#from typing import Any
#
#import yaml
#
#from context_based_anomalous_flow_detector.paths import CONFIG_PATH
#
##def load_config() -> dict[str, Any]:
##    #config_resource = files("context_based_anomalous_flow_detector") / "config.yaml"
##
##    #if not config_resource.exists():
##    #    raise FileNotFoundError(f"Config file not found at: {config_resource}")
##
##    #with open(config_resource, "r") as f:
##    #    return yaml.safe_load(f)
#
#
#def load_config() -> dict[str, Any]:
#    with open(CONFIG_PATH, "r") as f:
#        return yaml.safe_load(f)
#
#def get_dataset_params(dataset_id: str = "unsw_nb15") -> dict[str, Any]:
#    config = load_config()
#
#    if dataset_id not in config:
#        raise KeyError(f"Dataset '{dataset_id}' not defined in config.yaml")
#
#    params = config[dataset_id]
#
#    package_root = Path(files("context_based_anomalous_flow_detector")).resolve()
#    project_root = package_root.parent
#
#    if "file_path" in params:
#        params["file_path"] = (project_root / params["file_path"]).resolve()
#
#    return params