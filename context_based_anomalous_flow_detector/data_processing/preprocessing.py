"""Pure preprocessing transforms: canonical DataFrame -> model-input tensors."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from context_based_anomalous_flow_detector.schema import (
    CONT_COLS,
    CONT_PAD_VALUE,
    LABEL_PAD_VALUE,
    PROTO_PAD_VALUE,
    port_range_bucket,
    port_vocab_size,
)
pd.options.mode.copy_on_write = True


@dataclass
class PreprocessParams:
    """Configuration that controls how canonical data becomes model input.

    These parameters are user-controlled and recorded in the provenance manifest
    so training runs know exactly how the input was produced.
    """

    seq_len: int = 34
    group_cols: list[str] = field(default_factory=lambda: ["ipsrc"])
    time_col: str = "starttime"
    split: tuple[float, float] = (0.70, 0.85)
    # How many distinct destination ports keep their own class (see build_port_vocab/
    # quantize_port below); the coverage curve on unsw_nb15/v1 flattens fast (top 50
    # covers 68.2% of flows, top 500 only 72.3%), so a few dozen already capture most
    # of the head, and everything else falls back to a coarse IANA range bucket.
    port_top_k: int = 50
    # Scaling denominators (dataset-specific, kept here so they are part of config)
    log_div: dict[str, float] = field(
        default_factory=lambda: {
            "bytes_in": 20.0,
            "bytes_out": 20.0,
            "packets_in": 12.0,
            "packets_out": 12.0,
            "duration": 12.0,
            "time_delta": 15.0,
        }
    )
    ttl_max: float = 255.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "group_cols": self.group_cols,
            "time_col": self.time_col,
            "split": list(self.split),
            "log_div": self.log_div,
            "ttl_max": self.ttl_max,
            "port_top_k": self.port_top_k,
        }


def sort_flows(data: pd.DataFrame, params: PreprocessParams) -> pd.DataFrame:
    return data.sort_values(by=params.group_cols + [params.time_col]).reset_index(drop=True)


def featurize(sorted_df: pd.DataFrame, params: PreprocessParams) -> pd.DataFrame:
    """Log-scale numerics, derive tcp flag bits, and cast categoricals."""
    data = sorted_df.copy()

    data["time_delta"] = data.groupby(params.group_cols)[params.time_col].diff().fillna(0)

    for col, div in params.log_div.items():
        data[col] = np.log1p(data[col].astype(np.float32)) / div

    data["ttl"] = data["ttl"].astype(np.float32) / params.ttl_max

    flags_series = data["tcpflags"].fillna(0).astype(np.int64)
    data["tcp_fin"] = ((flags_series & 1) > 0).astype(np.float32)
    data["tcp_syn"] = ((flags_series & 2) > 0).astype(np.float32)
    data["tcp_rst"] = ((flags_series & 4) > 0).astype(np.float32)
    data["tcp_psh"] = ((flags_series & 8) > 0).astype(np.float32)
    data["tcp_ack"] = ((flags_series & 16) > 0).astype(np.float32)
    data["tcp_urg"] = ((flags_series & 32) > 0).astype(np.float32)

    data["portdst"] = data["portdst"].fillna(0).astype(np.int64)
    data["proto"] = data["proto"].fillna(0).astype(np.int64)

    return data


def build_port_vocab(train_df: pd.DataFrame, port_top_k: int) -> dict[int, int]:
    """Fit the compact port vocabulary on the train split ONLY -- fitting on eval/test
    too would leak their port-frequency distribution into what's supposed to be an
    unseen-data evaluation.

    Maps each of the `port_top_k` most frequent destination ports in `train_df` to its
    own class id (0..port_top_k-1, ranked by descending frequency; ties broken by port
    number for determinism). Ports not in this dict fall back to an IANA range bucket
    instead -- see quantize_port.
    """
    counts = train_df["portdst"].value_counts().rename("count").reset_index()
    counts.columns = ["port", "count"]
    counts = counts.sort_values(["count", "port"], ascending=[False, True])
    top_ports = counts["port"].head(port_top_k).tolist()
    return {int(port): rank for rank, port in enumerate(top_ports)}


def quantize_port(raw_port: pd.Series, port_vocab: dict[int, int], port_top_k: int) -> pd.Series:
    """Map raw destination ports (0-65535) to the compact vocabulary fit by
    build_port_vocab: the top-k most frequent ports keep their own class
    (0..port_top_k-1), everything else falls into one of the 3 IANA range buckets,
    placed right after the top-k classes (port_top_k, port_top_k+1, port_top_k+2)."""
    mapped = raw_port.map(port_vocab)
    unmapped = mapped.isna()
    if unmapped.any():
        buckets = raw_port[unmapped].map(port_range_bucket) + port_top_k
        mapped = mapped.where(~unmapped, buckets)
    return mapped.astype(np.int64)


def chronological_split(
    featurized: pd.DataFrame, params: PreprocessParams
) -> dict[str, pd.DataFrame]:
    """Split by a global time threshold across all hosts, not by row position.

    `featurized` is sorted by group_cols + [time_col] (host first, then time within
    host), so slicing by row position would split by host, not by time -- e.g. with
    group_cols=["ipsrc"], most or all of a given host's traffic could end up in a
    single split, and if attack traffic happens to come from hosts that sort earlier
    than params.split, eval/test could end up with zero attack flows. Instead, compute
    train/eval boundary timestamps from the global distribution of time_col, and assign
    each row by comparing its own timestamp to those thresholds -- so every split gets
    a temporal cross-section across every host.
    """
    train_frac, eval_frac = params.split
    time_col = params.time_col
    train_threshold, eval_threshold = featurized[time_col].quantile([train_frac, eval_frac])
    return {
        "train": featurized[featurized[time_col] <= train_threshold],
        "eval": featurized[
            (featurized[time_col] > train_threshold) & (featurized[time_col] <= eval_threshold)
        ],
        "test": featurized[featurized[time_col] > eval_threshold],
    }


def _build_sequence_tensors(
    df_slice: pd.DataFrame,
    params: PreprocessParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split per-group flows into masked sequences of seq_len.

    Returns (cont, proto, port, mask, label) where mask is PyTorch Transformer
    convention (True = pad/ignore). label is ground-truth benign/attack per
    flow, carried alongside for evaluation only -- it is never fed to the
    model as an input feature."""
    seq_len = params.seq_len
    group_cols = params.group_cols

    all_cont: list[torch.Tensor] = []
    all_proto: list[torch.Tensor] = []
    all_port: list[torch.Tensor] = []
    all_mask: list[torch.Tensor] = []
    all_label: list[torch.Tensor] = []

    for _, group_df in df_slice.groupby(group_cols):
        n = len(group_df)
        n_full = n // seq_len
        full_rows = n_full * seq_len
        left_overs = n % seq_len

        if n_full > 0:
            full_df = group_df.iloc[:full_rows]
            all_cont.append(
                torch.from_numpy(full_df[CONT_COLS].values.astype(np.float32)).reshape(
                    n_full, seq_len, len(CONT_COLS)
                )
            )
            all_proto.append(
                torch.from_numpy(full_df["proto"].values.astype(np.int64)).reshape(n_full, seq_len)
            )
            all_port.append(
                torch.from_numpy(full_df["portdst"].values.astype(np.int64)).reshape(
                    n_full, seq_len
                )
            )
            all_mask.append(torch.zeros((n_full, seq_len), dtype=torch.bool))
            all_label.append(
                torch.from_numpy(full_df["label"].values.astype(np.int64)).reshape(
                    n_full, seq_len
                )
            )

        if left_overs > 0:
            leftover_df = group_df.iloc[full_rows:]
            cont_pad = torch.full((seq_len, len(CONT_COLS)), CONT_PAD_VALUE, dtype=torch.float32)
            proto_pad = torch.full((seq_len,), PROTO_PAD_VALUE, dtype=torch.long)
            # One past the last real quantized port class (port_top_k explicit ports +
            # 3 range buckets), mirroring PROTO_PAD_VALUE's "index == class count" convention.
            port_pad_value = port_vocab_size(params.port_top_k)
            port_pad = torch.full((seq_len,), port_pad_value, dtype=torch.long)
            mask = torch.full((seq_len,), True, dtype=torch.bool)
            label_pad = torch.full((seq_len,), LABEL_PAD_VALUE, dtype=torch.long)

            cont_pad[:left_overs] = torch.from_numpy(
                leftover_df[CONT_COLS].values.astype(np.float32)
            )
            proto_pad[:left_overs] = torch.from_numpy(leftover_df["proto"].values.astype(np.int64))
            port_pad[:left_overs] = torch.from_numpy(
                leftover_df["portdst"].values.astype(np.int64)
            )
            mask[:left_overs] = False
            label_pad[:left_overs] = torch.from_numpy(
                leftover_df["label"].values.astype(np.int64)
            )

            all_cont.append(cont_pad.unsqueeze(0))
            all_proto.append(proto_pad.unsqueeze(0))
            all_port.append(port_pad.unsqueeze(0))
            all_mask.append(mask.unsqueeze(0))
            all_label.append(label_pad.unsqueeze(0))

    return (
        torch.cat(all_cont, dim=0),
        torch.cat(all_proto, dim=0),
        torch.cat(all_port, dim=0),
        torch.cat(all_mask, dim=0),
        torch.cat(all_label, dim=0),
    )


def preprocess_to_tensors(
    data: pd.DataFrame, params: PreprocessParams
) -> dict[str, dict[str, torch.Tensor]]:
    """Full transform pipeline: canonical data -> per-split model-input tensors."""
    sorted_df = sort_flows(data, params)
    featurized = featurize(sorted_df, params)
    splits = chronological_split(featurized, params)

    # Fit the port vocabulary on "train" only, then apply that same fitted mapping to
    # every split -- see build_port_vocab's docstring for why (avoids leaking eval/test
    # port frequencies into what should be an unseen-data evaluation).
    port_vocab = build_port_vocab(splits["train"], params.port_top_k)
    for split_df in splits.values():
        split_df["portdst"] = quantize_port(split_df["portdst"], port_vocab, params.port_top_k)

    return {
        name: dict(
            zip(
                ["cont", "proto", "port", "mask", "label"],
                _build_sequence_tensors(split_df, params),
            )
        )
        for name, split_df in splits.items()
    }
