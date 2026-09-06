"""PreprocessingHandler: reads canonical data, validates schema, produces model input."""

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from context_based_anomalous_flow_detector.data_processing._manifest import (
    read_manifest,
    write_manifest,
)
from context_based_anomalous_flow_detector.data_processing.preprocessing import (
    PreprocessParams,
    preprocess_to_tensors,
)
from context_based_anomalous_flow_detector.paths import BASE_DATA_PATH
from context_based_anomalous_flow_detector.schema import now_iso, validate_canonical


class PreprocessingHandler:
    """Reads canonical data, validates the schema, and creates model-input tensors."""

    def __init__(self, base_path: Path = BASE_DATA_PATH):
        self.base_path = Path(base_path)

    def run(
        self,
        dataset_id: str,
        version: str,
        params: PreprocessParams | dict[str, Any] | None = None,
        verify_hashes: bool = True,
    ) -> Path:
        """Preprocess canonical data for a given dataset_id/version.

        Reads canonical data, checks it against the expected schema, transforms it
        into model-input tensors, and saves them (with a provenance manifest) to
        base_path/dataset_id/version/preprocessed/.
        """
        canonical_dir = self.base_path / dataset_id / version / "canonical"
        canonical_path = canonical_dir / "canonical.parquet"
        if not canonical_path.exists():
            raise FileNotFoundError(f"Canonical data not found: {canonical_path}")

        canonical_manifest = read_manifest(canonical_dir)

        data = pd.read_parquet(canonical_path)
        validate_canonical(data)

        if verify_hashes:
            self._verify_against_manifest(data, canonical_manifest, dataset_id, version)

        if params is None:
            params = PreprocessParams()
        elif isinstance(params, dict):
            params = PreprocessParams(**params)

        tensors = preprocess_to_tensors(data, params)

        target_dir = self.base_path / dataset_id / version / "preprocessed"
        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, target_dir / "tensors.pt")

        write_manifest(
            target_dir,
            {
                "dataset_id": dataset_id,
                "version": version,
                "created_at": now_iso(),
                "canonical": {
                    "dataset_id": canonical_manifest.get("dataset_id"),
                    "version": canonical_manifest.get("version"),
                    "row_count": canonical_manifest.get("row_count"),
                    "output_file": canonical_manifest.get("output_file"),
                },
                "preprocess_params": params.to_dict(),
                "splits": {
                    name: {
                        "sequences": int(v["cont"].shape[0]),
                        "seq_len": int(v["cont"].shape[1]),
                        "cont_features": int(v["cont"].shape[2]),
                    }
                    for name, v in tensors.items()
                },
                "output_file": "tensors.pt",
            },
        )

        print(f"Preprocessed tensors saved to: {target_dir / 'tensors.pt'}")
        return target_dir

    @staticmethod
    def _verify_against_manifest(
        data: pd.DataFrame,
        manifest: dict[str, Any],
        dataset_id: str,
        version: str,
    ) -> None:
        if manifest.get("dataset_id") != dataset_id or manifest.get("version") != version:
            raise ValueError(
                f"Canonical manifest mismatch: expected {dataset_id}/{version}, "
                f"found {manifest.get('dataset_id')}/{manifest.get('version')}"
            )
        record_count = manifest.get("row_count")
        if record_count is not None and record_count != len(data):
            raise ValueError(
                f"Manifest row_count ({record_count}) does not match canonical rows ({len(data)})"
            )
        # NOTE: full per-column hash verification is available via schema.canonical_column_hashes;
        # computing it over the full dataset is expensive and thus skipped here by default.
