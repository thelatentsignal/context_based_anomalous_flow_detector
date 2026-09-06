from collections.abc import Callable
from pathlib import Path

import pandas as pd

from context_based_anomalous_flow_detector.data_processing._manifest import write_manifest
from context_based_anomalous_flow_detector.paths import BASE_DATA_PATH
from context_based_anomalous_flow_detector.schema import (
    canonical_column_hashes,
    now_iso,
    validate_canonical,
)


class CanonicalHandler:
    """Executes a datasource adapter and saves the canonical data to a hive-like path."""

    def __init__(self, base_path: Path = BASE_DATA_PATH):
        self.base_path = Path(base_path)

    def create(
        self,
        adapter: Callable[[str, str], pd.DataFrame],
        dataset_id: str,
        version: str,
    ) -> Path:
        """Save canonical data to base_path/dataset_id/version/canonical/."""
        data = adapter(dataset_id, version)

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Adapter must return a pd.DataFrame, got {type(data)}")

        # Validate against the canonical schema contract before persisting.
        validate_canonical(data)

        target_dir = self.base_path / dataset_id / version / "canonical"
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / "canonical.parquet"
        data.to_parquet(output_path, index=False)

        write_manifest(
            target_dir,
            {
                "dataset_id": dataset_id,
                "version": version,
                "created_at": now_iso(),
                "row_count": len(data),
                "columns": list(data.columns),
                "schema_hashes": canonical_column_hashes(data),
                "output_file": output_path.name,
            },
        )

        print(f"Canonical data saved to: {output_path}")
        return target_dir
