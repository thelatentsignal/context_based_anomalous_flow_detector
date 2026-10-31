import pandas as pd
import pytest

from context_based_anomalous_flow_detector.data_processing.adapters import unsw_nb15_adapter
from context_based_anomalous_flow_detector.schema import CANONICAL_COLUMN_NAMES, validate_canonical

RAW_COLUMNS = [
    "OUT_BYTES",
    "IN_BYTES",
    "OUT_PKTS",
    "IN_PKTS",
    "TCP_FLAGS",
    "MAX_TTL",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "FLOW_DURATION_MILLISECONDS",
    "FLOW_START_MILLISECONDS",
    "PROTOCOL",
    "Label",
    "Attack",
]


def _write_raw_csv(base_path, dataset_id, filename):
    raw_dir = base_path / dataset_id / "raw"
    raw_dir.mkdir(parents=True)
    rows = [
        [
            100,
            200,
            3,
            4,
            24,
            64,
            "10.0.0.1",
            "10.0.0.2",
            1234,
            80,
            15,
            1_600_000_000_000,
            6,
            0,
            "Benign",
        ],
        [
            5,
            0,
            1,
            0,
            2,
            128,
            "10.0.0.3",
            "10.0.0.2",
            4321,
            22,
            0,
            1_600_000_000_500,
            17,
            1,
            "Exploits",
        ],
    ]
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(raw_dir / filename, index=False)


def test_create_canonical_data_reads_given_file_and_returns_valid_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(unsw_nb15_adapter, "BASE_DATA_PATH", tmp_path)
    _write_raw_csv(tmp_path, "some_dataset", "tiny.csv")

    df = unsw_nb15_adapter.create_canonical_data("some_dataset", "v1", "tiny.csv")

    assert set(df.columns) == set(CANONICAL_COLUMN_NAMES)
    validate_canonical(df)
    assert len(df) == 2
    assert df.loc[1, "attack"] == "Exploits"
    assert df.loc[1, "portdst"] == 22


def test_create_canonical_data_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(unsw_nb15_adapter, "BASE_DATA_PATH", tmp_path)
    with pytest.raises(FileNotFoundError):
        unsw_nb15_adapter.create_canonical_data("some_dataset", "v1", "nope.csv")
