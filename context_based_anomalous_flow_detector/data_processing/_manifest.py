"""Shared helpers for writing provenance manifests beside data artifacts."""

from pathlib import Path
from typing import Any

import yaml


def write_manifest(
    target_dir: Path, metadata: dict[str, Any], filename: str = "metadata.yaml"
) -> Path:
    """Write a YAML manifest next to a data artifact for provenance tracking."""
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / filename
    with open(manifest_path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    return manifest_path


def read_manifest(target_dir: Path, filename: str = "metadata.yaml") -> dict[str, Any]:
    manifest_path = Path(target_dir) / filename
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r") as f:
        return yaml.safe_load(f) or {}
