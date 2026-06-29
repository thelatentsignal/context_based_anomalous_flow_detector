from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from context_based_anomalous_flow_detector.paths import MODELS_DIR


def create_run_dir(run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = MODELS_DIR / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_training_run(
    model: torch.nn.Module,
    config: dict[str, Any],
    metrics: dict[str, Any],
    run_dir: Path,
) -> None:
    torch.save(model.state_dict(), run_dir / "model.pt")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "model_file": "model.pt",
        "config_file": "config.yaml",
        "metrics_file": "metrics.json",
    }

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_model_state(run_dir: Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(run_dir / "model.pt", map_location=map_location)