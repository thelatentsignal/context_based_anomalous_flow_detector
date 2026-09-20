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
    CanonicalColumn("label", np.int16),
    CanonicalColumn("attack", str),
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
# Ground-truth label is carried alongside the model input for evaluation only
# (never fed to the model), so it needs its own pad sentinel too. -1 is safe
# because real labels are 0 (benign) / 1 (attack).
LABEL_PAD_VALUE: int = -1


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Port quantization
# ---------------------------------------------------------------------------
# Raw destination ports span 0-65535, but a per-component AUC breakdown of the
# anomaly detector showed that reconstructing the *exact* port is both the
# weakest signal of the three (cont/proto/port) and, worse, inversely
# correlated with attacks (many attacks repeatedly hit one target port, which
# becomes easy to predict from context, i.e. it's "camouflaged"). Despite that
# weak signal, port_dst_embedding + reconstruction_port made up 92.6% of the
# model's parameters (see modeling/anobert.py) -- almost the entire model
# existed to memorize a 65536-way categorical feature that barely helps.
#
# Instead of one class per raw port, only the `port_top_k` most frequent
# destination ports (fit on the train split, see preprocessing.build_port_vocab)
# get their own class; every other port collapses into one of these 3 IANA
# range buckets, keeping some information for the long tail instead of a
# single undifferentiated "other" class.
PORT_WELL_KNOWN_MAX = 1023  # IANA well-known ports: 0-1023
PORT_REGISTERED_MAX = 49151  # IANA registered ports: 1024-49151; above is dynamic/ephemeral
PORT_NUM_RANGE_BUCKETS = 3  # well-known / registered / dynamic-ephemeral


def port_range_bucket(raw_port: int) -> int:
    """Which of the 3 IANA range buckets a raw, un-quantized port falls into,
    as an offset (0/1/2) to be placed right after the port_top_k explicit
    per-port classes -- see preprocessing.quantize_port."""
    if raw_port <= PORT_WELL_KNOWN_MAX:
        return 0
    if raw_port <= PORT_REGISTERED_MAX:
        return 1
    return 2


def port_vocab_size(port_top_k: int) -> int:
    """Number of real (non-pad/mask) port classes for a given `port_top_k`:
    the top-k explicit ports plus the 3 range buckets for everything else.
    The pad/mask sentinel is one past this (same "index == count" convention
    as PROTO_PAD_VALUE = 256 for the 256 real proto classes 0-255)."""
    return port_top_k + PORT_NUM_RANGE_BUCKETS
