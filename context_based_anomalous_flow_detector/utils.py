from importlib.resources import files
from pathlib import Path
import yaml
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PACKAGE_ROOT / "context_based_anomalous_flow_detector" / "config.yaml"
def load_config()-> dict[str, Any]:
    """
    Dynamically locates the package root, reads the config.yaml,
    and handles automatic path extraction.
    """
    # 1. Locate and read the config file relative to the package
    config_resource = files("context_based_anomalous_flow_detector") / "config.yaml"

    if not config_resource.exists():
        raise FileNotFoundError(f"Config file not found at: {config_resource}")

    with open(config_resource, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_dataset_params(dataset_id: str = "unsw_nb15") -> dict[str, Any]:
    """
    Loads config and extracts dataset parameters with fully resolved paths.
    """
    config = load_config()

    if dataset_id not in config:
        raise KeyError(f"Dataset '{dataset_id}' not defined in config.yaml")

    params = config[dataset_id]

    # 2. Automatically resolve the relative path against your project root
    if "file_path" in params:
        package_root = Path(files("context_based_anomalous_flow_detector")).resolve()
        project_root = package_root.parent

        # Convert the string path from YAML into a clean, absolute Path object
        params["file_path"] = (project_root / params["file_path"]).resolve()

    return params


def save_model_safely(
        model: torch.nn.Module,
        config_dict: dict,
        base_name: str = 'model'
) -> Path:
    from datetime import datetime

    # Locate Cookiecutter's standard root 'models/' directory
    package_root = Path(files("context_based_anomalous_flow_detector")).resolve()
    project_root = package_root.parent
    output_dir = project_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 1. Save the PyTorch Weights
    model_dest = output_dir / f"{base_name}_{timestamp}.pt"
    torch.save(model.state_dict(), model_dest)

    # 2. Save the historical COPY of your configuration data
    config_dest = output_dir / f"{base_name}_{timestamp}_config.yaml"
    with open(config_dest, "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False)

    print(f"Saved snapshot to Cookiecutter models/ folder: {config_dest.name}")
    return model_dest
