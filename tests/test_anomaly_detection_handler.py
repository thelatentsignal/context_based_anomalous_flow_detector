from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from context_based_anomalous_flow_detector.data_processing._manifest import write_manifest
from context_based_anomalous_flow_detector.data_processing.preprocessing import (
    PreprocessParams,
    preprocess_to_tensors,
)
from context_based_anomalous_flow_detector.evaluation.anomaly_detection_handler import (
    AnomalyDetectionHandler,
    AnomalyDetectionParams,
)
from context_based_anomalous_flow_detector.modeling.anobert import AnoBert
from context_based_anomalous_flow_detector.schema import CONT_PAD_VALUE, PORT_PAD_VALUE, PROTO_PAD_VALUE
from context_based_anomalous_flow_detector.training.trainer_handler import TrainerHandler, TrainingParams

DATASET_ID = "synthetic"
VERSION = "v1"


def _tiny_model(seq_len: int = 6) -> AnoBert:
    torch.manual_seed(0)
    return AnoBert(num_continuous_features=13, d_model=16, nhead=2, num_layers=1, max_seq_len=seq_len)


def _naive_leave_one_out(model, cont_b, proto_b, port_b, mask_b, params) -> torch.Tensor:
    """Reference implementation: one forward pass per position, no vectorization trick.
    Used to cross-check the fast, diagonal-indexed AnomalyDetectionHandler._score_batch."""
    b, seq_len, _ = cont_b.shape
    scores = torch.zeros(b, seq_len)
    with torch.no_grad():
        for pos in range(seq_len):
            masked_cont = cont_b.clone()
            masked_proto = proto_b.clone()
            masked_port = port_b.clone()
            masked_cont[:, pos, :] = CONT_PAD_VALUE
            masked_proto[:, pos] = PROTO_PAD_VALUE
            masked_port[:, pos] = PORT_PAD_VALUE

            pred_cont, pred_proto, pred_port = model(
                masked_cont, masked_proto, masked_port, key_padding_mask=mask_b
            )

            cont_err = torch.mean((pred_cont[:, pos, :] - cont_b[:, pos, :]) ** 2, dim=-1)
            proto_err = nn.functional.cross_entropy(
                pred_proto[:, pos, :], proto_b[:, pos], reduction="none"
            )
            port_err = nn.functional.cross_entropy(
                pred_port[:, pos, :], port_b[:, pos], reduction="none"
            )
            scores[:, pos] = (
                params.cont_weight * cont_err
                + params.proto_weight * proto_err
                + params.port_weight * port_err
            )
    scores[mask_b] = 0.0
    return scores


def test_score_batch_matches_naive_per_position_loop():
    """The vectorized _score_batch replicates each sequence seq_len times and reads
    off a diagonal to get every leave-one-out prediction in one forward pass. That
    indexing trick is easy to get subtly wrong, so cross-check it against a plain,
    unvectorized per-position loop on a small, real (not random-init-only) model."""
    torch.manual_seed(0)
    seq_len = 6
    b = 3
    model = _tiny_model(seq_len)
    model.eval()

    cont_b = torch.randn(b, seq_len, 13)
    proto_b = torch.randint(0, 256, (b, seq_len))
    port_b = torch.randint(0, 65535, (b, seq_len))
    mask_b = torch.zeros(b, seq_len, dtype=torch.bool)
    mask_b[0, -2:] = True  # one sequence has trailing padding

    params = AnomalyDetectionParams(cont_weight=30.0, proto_weight=1.0, port_weight=0.5)

    fast = AnomalyDetectionHandler._score_batch(model, cont_b, proto_b, port_b, mask_b, params)
    naive = _naive_leave_one_out(model, cont_b, proto_b, port_b, mask_b, params)

    assert torch.allclose(fast, naive, atol=1e-5)


def test_score_batch_zeroes_out_padded_positions():
    seq_len = 5
    model = _tiny_model(seq_len)
    model.eval()

    cont_b = torch.randn(2, seq_len, 13)
    proto_b = torch.randint(0, 256, (2, seq_len))
    port_b = torch.randint(0, 65535, (2, seq_len))
    mask_b = torch.zeros(2, seq_len, dtype=torch.bool)
    mask_b[:, 3:] = True

    scores = AnomalyDetectionHandler._score_batch(
        model, cont_b, proto_b, port_b, mask_b, AnomalyDetectionParams()
    )
    assert torch.all(scores[mask_b] == 0.0)
    assert torch.all(scores[~mask_b] != 0.0)


def _write_synthetic_preprocessed(base_path: Path, n: int, seq_len: int, rng) -> None:
    df = pd.DataFrame(
        {
            "bytes_out": rng.integers(0, 1000, n),
            "bytes_in": rng.integers(0, 1000, n),
            "packets_out": rng.integers(0, 20, n),
            "packets_in": rng.integers(0, 20, n),
            "tcpflags": rng.integers(0, 64, n),
            "ttl": rng.integers(0, 255, n),
            "ipsrc": rng.choice(["1.1.1.1", "2.2.2.2"], n),
            "ipdst": "3.3.3.3",
            "portsrc": rng.integers(0, 65535, n),
            "portdst": rng.integers(0, 65535, n),
            "duration": rng.integers(0, 100, n),
            "starttime": np.sort(rng.integers(0, 1_000_000, n)),
            "proto": rng.integers(0, 10, n),
            "label": rng.integers(0, 2, n),
            "attack": "none",
        }
    )
    params = PreprocessParams(seq_len=seq_len)
    tensors = preprocess_to_tensors(df, params)
    preprocessed_dir = base_path / DATASET_ID / VERSION / "preprocessed"
    preprocessed_dir.mkdir(parents=True)
    torch.save(tensors, preprocessed_dir / "tensors.pt")
    write_manifest(
        preprocessed_dir,
        {"dataset_id": DATASET_ID, "version": VERSION, "preprocess_params": params.to_dict()},
    )


@pytest.fixture
def detector_and_dirs(tmp_path: Path):
    base_path = tmp_path / "data"
    models_root = tmp_path / "models"
    rng = np.random.default_rng(0)
    _write_synthetic_preprocessed(base_path, n=300, seq_len=6, rng=rng)

    trainer = TrainerHandler(base_path=base_path, models_root=models_root)
    trainer.run(
        DATASET_ID,
        VERSION,
        params=TrainingParams(seed=1, batch_size=4, num_epochs=2, d_model=16, n_heads=2, num_layers=1),
    )
    detector = AnomalyDetectionHandler(base_path=base_path, models_root=models_root)
    return detector, models_root


def test_run_writes_results_with_expected_shapes(detector_and_dirs):
    detector, models_root = detector_and_dirs
    model_dir = detector.run(DATASET_ID, VERSION, split="test", params=AnomalyDetectionParams(batch_size=4))

    assert (model_dir / "anomaly_detection.json").exists()
    assert (model_dir / "anomaly_scores.pt").exists()

    blob = torch.load(model_dir / "anomaly_scores.pt", weights_only=True)
    assert blob["scores"].shape == blob["labels"].shape == blob["mask"].shape
    # Padded positions must score exactly 0.
    assert torch.all(blob["scores"][blob["mask"]] == 0.0)


def test_metrics_omit_auc_when_only_one_class_present(tmp_path: Path):
    base_path = tmp_path / "data"
    models_root = tmp_path / "models"
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame(
        {
            "bytes_out": rng.integers(0, 1000, n),
            "bytes_in": rng.integers(0, 1000, n),
            "packets_out": rng.integers(0, 20, n),
            "packets_in": rng.integers(0, 20, n),
            "tcpflags": rng.integers(0, 64, n),
            "ttl": rng.integers(0, 255, n),
            "ipsrc": rng.choice(["1.1.1.1", "2.2.2.2"], n),
            "ipdst": "3.3.3.3",
            "portsrc": rng.integers(0, 65535, n),
            "portdst": rng.integers(0, 65535, n),
            "duration": rng.integers(0, 100, n),
            "starttime": np.sort(rng.integers(0, 1_000_000, n)),
            "proto": rng.integers(0, 10, n),
            "label": 0,  # single class only
            "attack": "none",
        }
    )
    params = PreprocessParams(seq_len=6)
    tensors = preprocess_to_tensors(df, params)
    preprocessed_dir = base_path / DATASET_ID / VERSION / "preprocessed"
    preprocessed_dir.mkdir(parents=True)
    torch.save(tensors, preprocessed_dir / "tensors.pt")
    write_manifest(
        preprocessed_dir,
        {"dataset_id": DATASET_ID, "version": VERSION, "preprocess_params": params.to_dict()},
    )

    trainer = TrainerHandler(base_path=base_path, models_root=models_root)
    trainer.run(
        DATASET_ID,
        VERSION,
        params=TrainingParams(seed=1, batch_size=4, num_epochs=1, d_model=16, n_heads=2, num_layers=1),
    )
    detector = AnomalyDetectionHandler(base_path=base_path, models_root=models_root)
    model_dir = detector.run(DATASET_ID, VERSION, split="test", params=AnomalyDetectionParams(batch_size=4))

    import json

    with open(model_dir / "anomaly_detection.json") as f:
        results = json.load(f)
    assert "roc_auc" not in results["metrics"]
    assert results["metrics"]["n_attack"] == 0
