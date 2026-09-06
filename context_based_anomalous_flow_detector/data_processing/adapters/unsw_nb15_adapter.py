import numpy as np
import pandas as pd

from context_based_anomalous_flow_detector.paths import PROJECT_ROOT
from context_based_anomalous_flow_detector.schema import CANONICAL_COLUMNS


def create_canonical_data(dataset_id: str, version: str) -> pd.DataFrame:
    """Read raw UNSW-NB15 data and return canonical format."""
    raw_path = PROJECT_ROOT / "data" / dataset_id / "raw" / "NF-UNSW-NB15-v3.csv"
    raw_data = pd.read_csv(raw_path)

    cols = [
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
    ]

    res_df = raw_data[cols].copy()
    res_df.rename(
        columns={
            "OUT_BYTES": "bytes_out",
            "IN_BYTES": "bytes_in",
            "OUT_PKTS": "packets_out",
            "IN_PKTS": "packets_in",
            "MAX_TTL": "ttl",
            "PROTOCOL": "proto",
            "IPV4_SRC_ADDR": "ipsrc",
            "IPV4_DST_ADDR": "ipdst",
            "L4_SRC_PORT": "portsrc",
            "L4_DST_PORT": "portdst",
            "FLOW_DURATION_MILLISECONDS": "duration",
            "FLOW_START_MILLISECONDS": "starttime",
            "TCP_FLAGS": "tcpflags",
        },
        inplace=True,
    )

    return cast_to_canonical_dtypes(res_df)


def cast_to_canonical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to the canonical (small) dtypes from the schema."""
    res = df.copy()
    for col in CANONICAL_COLUMNS:
        try:
            dt = np.dtype(col.dtype)  # raises for str/object (non-numpy) dtypes
        except TypeError:
            continue
        res[col.name] = res[col.name].astype(dt)
    return res
