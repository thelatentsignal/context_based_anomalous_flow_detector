"""Central schema contracts for the canonical data and the model input.

Single source of truth for what the pipeline expects at each stage:
- CANONICAL_SCHEMA: what the canonical (stage-1) output must contain.
- CONT_COLS: which canonical columns feed the continuous stream.
- Preprocessing params: how canonical data is transformed into model input.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical schema (stage 1 output)
# ---------------------------------------------------------------------------
# The canonical contract: every stage-1 datasource adapter must return a
# DataFrame with (at least) these columns carrying the expected dtypes.
@dataclass(frozen=True)
class CanonicalColumn:
    name: str
    dtype: np.dtype | type


CANONICAL_COLUMNS: list[CanonicalColumn] = [
    CanonicalColumn("bytes_out", np.int32),
    CanonicalColumn("bytes_in", np.int32),
    CanonicalColumn("packets_out", np.int32),
    CanonicalColumn("packets_in", np.int32),
    CanonicalColumn("tcpflags", np.int8),
    CanonicalColumn("ttl", np.int16),
    CanonicalColumn("ipsrc", str),  # IPv4 address
    CanonicalColumn("ipdst", str),  # IPv4 address
    CanonicalColumn("portsrc", np.int32),
    CanonicalColumn("portdst", np.int32),
    CanonicalColumn("duration", np.int32),
    CanonicalColumn("starttime", np.int64),
    CanonicalColumn("proto", np.int16),
]

CANONICAL_COLUMN_NAMES: list[str] = [c.name for c in CANONICAL_COLUMNS]


def validate_canonical(df: pd.DataFrame) -> None:
    """Raise if df does not match the canonical schema (columns + dtypes)."""
    expected = {c.name: c.dtype for c in CANONICAL_COLUMNS}

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Canonical data is missing required columns: {missing}")

    for name, expected_dtype in expected.items():
        actual_dtype = df[name].dtype
        if not _dtype_matches(actual_dtype, expected_dtype):
            raise TypeError(f"Column '{name}' has dtype {actual_dtype}, expected {expected_dtype}")


def _dtype_matches(actual: Any, expected: np.dtype | type) -> bool:
    """Check whether a pandas column dtype matches the expected numpy/Python type."""
    actual_py = pd.api.types.pandas_dtype(actual).type

    if expected is str:
        # Strings are stored as pandas 'str', numpy unicode, or generic 'object'.
        return actual_py in (str, np.str_, object, np.object_)
    if expected is object:
        return actual_py in (object, np.object_)

    expected_np = np.dtype(expected)
    try:
        return np.issubdtype(np.dtype(actual), expected_np)
    except TypeError:
        return actual_py is expected_np.type


def canonical_column_hashes(df: pd.DataFrame) -> dict[str, str]:
    """Return a per-column dtype + hash summary used for provenance drift checks."""
    return {c.name: _digest_col(df[c.name]) for c in CANONICAL_COLUMNS}


def _digest_col(series: pd.Series) -> str:
    import hashlib

    m = hashlib.sha256()
    m.update(str(series.dtype).encode())
    m.update(series.astype(str).str.cat(sep="\x1f").encode("utf-8", errors="ignore"))
    return m.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Continuous features stream (after featurization)
# ---------------------------------------------------------------------------
CONT_COLS: list[str] = [
    "bytes_in",
    "bytes_out",
    "packets_in",
    "packets_out",
    "ttl",
    "duration",
    "time_delta",
    "tcp_fin",
    "tcp_syn",
    "tcp_rst",
    "tcp_psh",
    "tcp_ack",
    "tcp_urg",
]

# Padding/sentinel values used for sequence masking
CONT_PAD_VALUE: float = -1.0
PROTO_PAD_VALUE: int = 256
PORT_PAD_VALUE: int = 65536


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
