import numpy as np
import pandas as pd
import pytest

from context_based_anomalous_flow_detector.schema import (
    CANONICAL_COLUMN_NAMES,
    validate_canonical,
)


def _valid_canonical_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 20
    return pd.DataFrame(
        {
            "bytes_out": rng.integers(0, 2**31 - 1, n, dtype=np.int32),
            "bytes_in": rng.integers(0, 2**31 - 1, n, dtype=np.int32),
            "packets_out": rng.integers(0, 2**31 - 1, n, dtype=np.int32),
            "packets_in": rng.integers(0, 2**31 - 1, n, dtype=np.int32),
            "tcpflags": rng.integers(0, 64, n, dtype=np.int8),
            "ttl": rng.integers(0, 255, n, dtype=np.int16),
            "ipsrc": ["10.0.0.1"] * n,
            "ipdst": ["10.0.0.2"] * n,
            "portsrc": rng.integers(0, 65535, n, dtype=np.int32),
            "portdst": rng.integers(0, 65535, n, dtype=np.int32),
            "duration": rng.integers(0, 10**6, n, dtype=np.int32),
            "starttime": rng.integers(0, 2**63, n, dtype=np.int64),
            "proto": rng.integers(1, 17, n, dtype=np.int16),
            "label": rng.integers(0, 2, n, dtype=np.int16),
            "attack": ["Benign"] * n,
        }
    )


def test_validate_canonical_accepts_valid_frame():
    df = _valid_canonical_df()
    validate_canonical(df)


def test_validate_canonical_rejects_missing_column():
    df = _valid_canonical_df().drop(columns=["proto"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_canonical(df)


def test_validate_canonical_rejects_wrong_dtype():
    df = _valid_canonical_df()
    df["ttl"] = df["ttl"].astype(np.float64)
    with pytest.raises(TypeError):
        validate_canonical(df)


def test_canonical_schema_has_expected_columns():
    assert set(CANONICAL_COLUMN_NAMES) == {
        "bytes_out",
        "bytes_in",
        "packets_out",
        "packets_in",
        "tcpflags",
        "ttl",
        "ipsrc",
        "ipdst",
        "portsrc",
        "portdst",
        "duration",
        "starttime",
        "proto",
        "label",
        "attack",
    }