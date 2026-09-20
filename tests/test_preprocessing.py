import numpy as np
import pandas as pd
import pytest
import torch

from context_based_anomalous_flow_detector.data_processing.preprocessing import (
    PreprocessParams,
    _build_sequence_tensors,
    build_port_vocab,
    chronological_split,
    featurize,
    preprocess_to_tensors,
    quantize_port,
    sort_flows,
)
from context_based_anomalous_flow_detector.schema import (
    CONT_PAD_VALUE,
    LABEL_PAD_VALUE,
    PORT_REGISTERED_MAX,
    PORT_WELL_KNOWN_MAX,
    PROTO_PAD_VALUE,
    port_vocab_size,
)


def _flow(ipsrc, starttime, proto=6, portdst=80, tcpflags=0, label=0, **overrides):
    row = {
        "bytes_out": 100,
        "bytes_in": 100,
        "packets_out": 2,
        "packets_in": 2,
        "tcpflags": tcpflags,
        "ttl": 64,
        "ipsrc": ipsrc,
        "ipdst": "10.0.0.1",
        "portsrc": 12345,
        "portdst": portdst,
        "duration": 1,
        "starttime": starttime,
        "proto": proto,
        "label": label,
        "attack": "Benign" if label == 0 else "Attack",
    }
    row.update(overrides)
    return row


def _params(**overrides) -> PreprocessParams:
    defaults = dict(seq_len=3, group_cols=["ipsrc"], time_col="starttime", split=(0.70, 0.85))
    defaults.update(overrides)
    return PreprocessParams(**defaults)


def test_sort_flows_orders_by_group_then_time():
    df = pd.DataFrame(
        [
            _flow("B", 5),
            _flow("A", 20),
            _flow("A", 10),
            _flow("B", 1),
        ]
    )
    sorted_df = sort_flows(df, _params())
    assert list(zip(sorted_df["ipsrc"], sorted_df["starttime"])) == [
        ("A", 10),
        ("A", 20),
        ("B", 1),
        ("B", 5),
    ]


def test_featurize_computes_time_delta_per_group():
    df = pd.DataFrame([_flow("A", 10), _flow("A", 25), _flow("B", 100)])
    sorted_df = sort_flows(df, _params())
    featurized = featurize(sorted_df, _params())

    # First row of each group has no predecessor -> time_delta 0.
    a_rows = featurized[featurized["ipsrc"] == "A"]
    assert a_rows["time_delta"].tolist()[0] == 0
    assert a_rows["time_delta"].tolist()[1] == pytest.approx(np.log1p(15) / 15.0)

    b_rows = featurized[featurized["ipsrc"] == "B"]
    assert b_rows["time_delta"].tolist()[0] == 0


def test_featurize_derives_tcp_flag_bits():
    # SYN (2) + ACK (16) = 18
    df = pd.DataFrame([_flow("A", 1, tcpflags=18)])
    featurized = featurize(sort_flows(df, _params()), _params())
    row = featurized.iloc[0]
    assert row["tcp_syn"] == 1.0
    assert row["tcp_ack"] == 1.0
    assert row["tcp_fin"] == 0.0
    assert row["tcp_rst"] == 0.0
    assert row["tcp_psh"] == 0.0
    assert row["tcp_urg"] == 0.0


def test_featurize_preserves_label_column():
    df = pd.DataFrame([_flow("A", 1, label=1), _flow("A", 2, label=0)])
    featurized = featurize(sort_flows(df, _params()), _params())
    assert featurized["label"].tolist() == [1, 0]


def test_chronological_split_uses_global_time_not_row_position():
    """Regression test for the split bug: sorting by ipsrc-then-time and slicing
    by row position put whole hosts on one side of the split, so a host's attack
    traffic could be entirely excluded from eval/test. The fix compares each row's
    own timestamp against a global threshold, so a single host's flows can straddle
    multiple splits -- mirroring the whiteboard example (host A: 8:25/8:26/8:27,
    host B: 7:04/7:05/8:36, split at 8:26)."""
    df = pd.DataFrame(
        [
            _flow("A", 825),
            _flow("A", 826),
            _flow("A", 827),
            _flow("B", 704),
            _flow("B", 705),
            _flow("B", 836),
        ]
    )
    featurized = featurize(sort_flows(df, _params()), _params())
    # A single 70/30-style split: everything <= the 826 threshold is "train".
    splits = chronological_split(featurized, _params(split=(0.6666, 0.6666)))

    train_pairs = set(zip(splits["train"]["ipsrc"], splits["train"]["starttime"]))
    test_pairs = set(zip(splits["test"]["ipsrc"], splits["test"]["starttime"]))

    # Both hosts contribute to the early split (train)...
    assert ("A", 825) in train_pairs
    assert ("A", 826) in train_pairs
    assert ("B", 704) in train_pairs
    assert ("B", 705) in train_pairs
    # ...and both hosts also contribute to the late split (test) --
    # i.e. host B is *not* wholly excluded just because its block sorts first.
    assert ("A", 827) in test_pairs
    assert ("B", 836) in test_pairs


def test_chronological_split_row_counts_sum_to_total():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        [_flow(f"host{i % 5}", int(t)) for i, t in enumerate(rng.integers(0, 100_000, 200))]
    )
    featurized = featurize(sort_flows(df, _params()), _params())
    splits = chronological_split(featurized, _params())
    assert sum(len(s) for s in splits.values()) == len(featurized)


def test_build_sequence_tensors_full_multiple_of_seq_len():
    seq_len = 3
    df = pd.DataFrame([_flow("A", t, portdst=1000 + t, proto=t % 5) for t in range(6)])
    params = _params(seq_len=seq_len)
    featurized = featurize(sort_flows(df, params), params)

    cont, proto, port, mask, label = _build_sequence_tensors(featurized, params)

    assert cont.shape == (2, seq_len, cont.shape[2])
    assert proto.shape == (2, seq_len)
    assert port.shape == (2, seq_len)
    assert mask.shape == (2, seq_len)
    assert label.shape == (2, seq_len)
    assert not mask.any()  # no padding needed, exact multiple of seq_len


def test_build_sequence_tensors_pads_leftover_with_sentinels():
    seq_len = 4
    df = pd.DataFrame(
        [_flow("A", t, portdst=1000 + t, proto=1, label=1 if t == 0 else 0) for t in range(2)]
    )
    params = _params(seq_len=seq_len)
    featurized = featurize(sort_flows(df, params), params)

    cont, proto, port, mask, label = _build_sequence_tensors(featurized, params)

    assert cont.shape == (1, seq_len, cont.shape[2])
    # First 2 positions real, last 2 padded.
    assert mask.tolist() == [[False, False, True, True]]
    assert proto[0, :2].tolist() == [1, 1]
    assert proto[0, 2:].tolist() == [PROTO_PAD_VALUE, PROTO_PAD_VALUE]
    expected_port_pad = port_vocab_size(params.port_top_k)
    assert port[0, 2:].tolist() == [expected_port_pad, expected_port_pad]
    assert torch.allclose(cont[0, 2:], torch.full((2, cont.shape[2]), CONT_PAD_VALUE))
    assert label[0, :2].tolist() == [1, 0]
    assert label[0, 2:].tolist() == [LABEL_PAD_VALUE, LABEL_PAD_VALUE]


def test_build_sequence_tensors_label_stays_aligned_with_cont():
    """label[i, j] must describe the same flow as cont[i, j] -- verify by encoding
    the flow's own label into a continuous feature (bytes_in, CONT_COLS[0]) and
    checking they match position-for-position across a full sequence plus a padded
    leftover."""
    seq_len = 3
    rows = []
    for i, t in enumerate(range(7)):
        label = i % 2
        # bytes_in encodes the label so we can cross-check alignment after scaling.
        rows.append(_flow("A", t, label=label, bytes_in=1 if label else 0))
    df = pd.DataFrame(rows)
    params = _params(seq_len=seq_len)
    featurized = featurize(sort_flows(df, params), params)

    cont, _, _, mask, label = _build_sequence_tensors(featurized, params)
    real = ~mask
    bytes_in_col = 0  # first entry of CONT_COLS
    encoded_label = (cont[..., bytes_in_col] > 0).long()
    assert torch.equal(encoded_label[real], label[real])


def test_build_port_vocab_ranks_by_frequency_then_port_number():
    """Top-k ports are picked by descending frequency; equal-frequency ports are
    ordered by port number so the mapping is deterministic run to run."""
    # port 80 appears 3x, port 443 appears 2x, ports 22/53 appear 1x each (tied).
    train_df = pd.DataFrame(
        {"portdst": [80, 80, 80, 443, 443, 22, 53]}
    )
    vocab = build_port_vocab(train_df, port_top_k=3)

    assert vocab == {80: 0, 443: 1, 22: 2}  # tie between 22/53 broken by port number


def test_quantize_port_keeps_top_k_ports_as_own_class():
    vocab = {80: 0, 443: 1}
    quantized = quantize_port(pd.Series([80, 443, 80]), vocab, port_top_k=2)
    assert quantized.tolist() == [0, 1, 0]


def test_quantize_port_buckets_ports_outside_top_k_by_iana_range():
    """A port not in the fitted vocab falls back to one of the 3 IANA range buckets,
    placed right after the port_top_k explicit classes (port_top_k + 0/1/2)."""
    vocab = {80: 0}  # only port 80 got an explicit class
    port_top_k = 1
    raw_ports = pd.Series(
        [
            80,  # explicit class
            PORT_WELL_KNOWN_MAX,  # top of well-known range -> bucket 0
            PORT_WELL_KNOWN_MAX + 1,  # bottom of registered range -> bucket 1
            PORT_REGISTERED_MAX,  # top of registered range -> bucket 1
            PORT_REGISTERED_MAX + 1,  # bottom of dynamic/ephemeral range -> bucket 2
        ]
    )
    quantized = quantize_port(raw_ports, vocab, port_top_k)
    assert quantized.tolist() == [0, port_top_k + 0, port_top_k + 1, port_top_k + 1, port_top_k + 2]


def test_preprocess_to_tensors_fits_port_vocab_on_train_split_only():
    """Regression guard against leakage: a port that's frequent in eval/test but
    absent from train must NOT get its own class -- it should fall back to an IANA
    range bucket like any other port train never saw."""
    ephemeral_port = 60000  # > PORT_REGISTERED_MAX (49151) -> dynamic/ephemeral bucket
    # Chronological split puts early timestamps in train; ephemeral_port only ever
    # appears late (test split), port 80 only appears early (train split).
    rows = [_flow("A", t, portdst=80) for t in range(70)]
    rows += [_flow("A", t, portdst=ephemeral_port) for t in range(70, 100)]
    df = pd.DataFrame(rows)
    params = _params(seq_len=100, port_top_k=5)  # 1 host group -> whole df as one sequence
    result = preprocess_to_tensors(df, params)

    # ephemeral_port never appeared in train, so it must land in the dynamic/ephemeral
    # range bucket (port_top_k + 2), not get a dedicated top-k class.
    test_ports = result["test"]["port"][~result["test"]["mask"]]
    assert test_ports.numel() > 0
    assert torch.all(test_ports == params.port_top_k + 2)


def test_preprocess_to_tensors_end_to_end_shape_and_keys():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        [
            _flow(
                f"host{i % 4}",
                int(t),
                portdst=int(rng.integers(0, 65535)),
                proto=int(rng.integers(0, 10)),
                label=int(rng.integers(0, 2)),
            )
            for i, t in enumerate(np.sort(rng.integers(0, 100_000, n)))
        ]
    )
    params = _params(seq_len=5)
    result = preprocess_to_tensors(df, params)

    assert set(result.keys()) == {"train", "eval", "test"}
    for split_name, tensors in result.items():
        assert set(tensors.keys()) == {"cont", "proto", "port", "mask", "label"}
        n_seq = tensors["cont"].shape[0]
        for key in ("proto", "port", "mask", "label"):
            assert tensors[key].shape == (n_seq, params.seq_len)
