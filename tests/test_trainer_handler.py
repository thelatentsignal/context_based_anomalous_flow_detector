from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from context_based_anomalous_flow_detector.data_processing._manifest import write_manifest
from context_based_anomalous_flow_detector.data_processing.preprocessing import (
    PreprocessParams,
    preprocess_to_tensors,
)
from context_based_anomalous_flow_detector.training import (
    trainer_handler as trainer_handler_module,
)
from context_based_anomalous_flow_detector.training.trainer_handler import (
    TrainerHandler,
    TrainingParams,
    load_trained_model,
)

DATASET_ID = "synthetic"
VERSION = "v1"


def _write_synthetic_preprocessed(base_path: Path, n: int = 200, seq_len: int = 5) -> None:
    rng = np.random.default_rng(0)
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


def _tiny_training_params(**overrides) -> TrainingParams:
    defaults = dict(seed=1, batch_size=4, num_epochs=3, d_model=16, n_heads=2, num_layers=1)
    defaults.update(overrides)
    return TrainingParams(**defaults)


@pytest.fixture
def trainer(tmp_path: Path) -> TrainerHandler:
    base_path = tmp_path / "data"
    models_root = tmp_path / "models"
    _write_synthetic_preprocessed(base_path)
    return TrainerHandler(base_path=base_path, models_root=models_root)


def test_training_params_roundtrip_through_dict():
    params = TrainingParams(
        seed=7,
        batch_size=32,
        num_epochs=10,
        num_cont_features=13,
        d_model=64,
        n_heads=8,
        num_layers=3,
    )
    restored = TrainingParams.from_dict(params.to_dict())
    assert restored == params


def test_run_creates_model_and_manifest(trainer: TrainerHandler):
    model_dir = trainer.run(DATASET_ID, VERSION, params=_tiny_training_params())

    assert (model_dir / "model.pt").exists()
    assert (model_dir / "metadata.yaml").exists()

    model, manifest = load_trained_model(DATASET_ID, VERSION, models_root=trainer.models_root)
    metrics = manifest["metrics"]
    assert "best_eval_loss" in metrics
    assert "best_epoch" in metrics
    assert metrics["stopped_epoch"] >= metrics["best_epoch"] > 0


def test_resume_requires_existing_checkpoint(trainer: TrainerHandler):
    with pytest.raises(FileNotFoundError):
        trainer.run(DATASET_ID, VERSION, params=_tiny_training_params(), resume=True)


def test_resume_never_regresses_saved_best(trainer: TrainerHandler):
    trainer.run(DATASET_ID, VERSION, params=_tiny_training_params(num_epochs=2))
    _, manifest_before = load_trained_model(DATASET_ID, VERSION, models_root=trainer.models_root)
    best_before = manifest_before["metrics"]["best_eval_loss"]

    # A generous num_epochs so the run has a real chance to (but need not) improve --
    # the invariant under test is that it can never end up worse than best_before,
    # regardless of whether these new epochs help.
    trainer.run(DATASET_ID, VERSION, params=_tiny_training_params(num_epochs=5), resume=True)
    _, manifest_after = load_trained_model(DATASET_ID, VERSION, models_root=trainer.models_root)
    best_after = manifest_after["metrics"]["best_eval_loss"]

    assert best_after <= best_before + 1e-9


def test_resume_continues_epoch_numbering_without_collision(trainer: TrainerHandler):
    trainer.run(DATASET_ID, VERSION, params=_tiny_training_params(num_epochs=2))
    _, manifest_before = load_trained_model(DATASET_ID, VERSION, models_root=trainer.models_root)
    stopped_before = manifest_before["metrics"]["stopped_epoch"]

    trainer.run(DATASET_ID, VERSION, params=_tiny_training_params(num_epochs=2), resume=True)
    _, manifest_after = load_trained_model(DATASET_ID, VERSION, models_root=trainer.models_root)
    metrics_after = manifest_after["metrics"]

    # Every epoch key from before the resume is still present (history preserved)...
    for key in manifest_before["metrics"]:
        assert key in metrics_after
    # ...and new epoch keys start strictly after where the previous run stopped.
    assert f"epoch_{stopped_before + 1}_train_loss" in metrics_after
    assert metrics_after["stopped_epoch"] > stopped_before


def test_resume_uses_a_different_seed_each_call(trainer: TrainerHandler, monkeypatch):
    """Regression test: resuming used to reseed with the same fixed value every time,
    so repeated resumes from a plateaued checkpoint deterministically replayed the
    exact same (failing) epochs forever. The seed must vary with epoch_offset."""
    seeds_used = []
    real_manual_seed = torch.manual_seed

    def recording_manual_seed(seed):
        seeds_used.append(seed)
        return real_manual_seed(seed)

    monkeypatch.setattr(trainer_handler_module.torch, "manual_seed", recording_manual_seed)

    trainer.run(DATASET_ID, VERSION, params=_tiny_training_params(num_epochs=2))
    trainer.run(DATASET_ID, VERSION, params=_tiny_training_params(num_epochs=2), resume=True)
    trainer.run(DATASET_ID, VERSION, params=_tiny_training_params(num_epochs=2), resume=True)

    assert len(seeds_used) == 3
    assert len(set(seeds_used)) == 3  # every call got a distinct seed


def test_load_trained_model_reconstructs_matching_weights(trainer: TrainerHandler):
    trainer.run(DATASET_ID, VERSION, params=_tiny_training_params())
    model, _ = load_trained_model(DATASET_ID, VERSION, models_root=trainer.models_root)

    state_dict = torch.load(
        trainer.models_root / DATASET_ID / VERSION / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    for key, value in state_dict.items():
        assert torch.equal(model.state_dict()[key], value)
