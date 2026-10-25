"""TrainerHandler: reads preprocessed tensors and trains an AnoBert model.

Saves the trained model artifact and a provenance manifest to
<models_root>/<dataset_id>/<version>/.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from context_based_anomalous_flow_detector.data_processing._manifest import (
    read_manifest,
    write_manifest,
)
from context_based_anomalous_flow_detector.modeling.anobert import AnoBert
from context_based_anomalous_flow_detector.paths import BASE_DATA_PATH, TRAINED_MODELS_DIR
from context_based_anomalous_flow_detector.schema import now_iso, port_vocab_size
from context_based_anomalous_flow_detector.training.train import train_model


@dataclass
class TrainingParams:
    """Configuration that controls how the AnoBert model is trained.

    Recorded in the training manifest so runs are reproducible.
    """

    seed: int = 42
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Model architecture
    num_cont_features: int = 13
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "device": self.device,
            "model": {
                "name": "anobert",
                "num_cont_features": self.num_cont_features,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "num_layers": self.num_layers,
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingParams":
        """Reconstruct params from the nested form produced by `to_dict` (e.g. read back from metadata.yaml)."""
        model = d.get("model", {})
        return cls(
            seed=d["seed"],
            batch_size=d["batch_size"],
            num_epochs=d["num_epochs"],
            learning_rate=d["learning_rate"],
            weight_decay=d["weight_decay"],
            device=d["device"],
            num_cont_features=model["num_cont_features"],
            d_model=model["d_model"],
            n_heads=model["n_heads"],
            num_layers=model["num_layers"],
        )


class TrainerHandler:
    def __init__(self, base_path: Path = BASE_DATA_PATH, models_root: Path = TRAINED_MODELS_DIR):
        self.base_path = Path(base_path)
        self.models_root = Path(models_root)

    def run(self, dataset_id: str, version: str, params=None, resume: bool = False) -> Path:
        """Train (or continue training) a model for dataset_id/version.

        If `resume=True`, warm-starts from the existing model.pt at
        <models_root>/<dataset_id>/<version>/, continues epoch numbering from
        where that checkpoint's metrics left off, and merges the new metrics
        into the existing ones instead of overwriting the history.
        """
        params = self._coerce_params(params)

        preprocessed_dir = self.base_path / dataset_id / version / "preprocessed"
        tensors_path = preprocessed_dir / "tensors.pt"
        if not tensors_path.exists():
            raise FileNotFoundError(f"Preprocessed tensors not found: {tensors_path}.")

        preprocess_manifest = read_manifest(preprocessed_dir)
        target_dir = self.models_root / dataset_id / version

        initial_state_dict = None
        initial_best_eval_loss = float("inf")
        initial_best_epoch = 0
        epoch_offset = 0
        previous_metrics: dict[str, Any] = {}

        if resume:
            checkpoint_path = target_dir / "model.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"No existing checkpoint to resume from: {checkpoint_path}"
                )
            previous_metrics = read_manifest(target_dir).get("metrics", {})
            epoch_offset = previous_metrics.get("stopped_epoch", 0)
            initial_best_eval_loss = previous_metrics.get("best_eval_loss", float("inf"))
            # older metadata.yaml files predate best_epoch; fall back to stopped_epoch
            initial_best_epoch = previous_metrics.get("best_epoch", epoch_offset)
            initial_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            print(
                f"Resuming from epoch {epoch_offset} "
                f"(best eval loss so far: {initial_best_eval_loss:.6f}, from epoch {initial_best_epoch})"
            )

        # Offsetting the seed by epoch_offset means a resumed run explores a fresh
        # shuffle/masking trajectory instead of deterministically replaying the exact
        # same epochs every time resume=True is called from the same checkpoint.
        torch.manual_seed(params.seed + epoch_offset)
        device = self._resolve_device(params.device)
        print(f"Using hardware accelerator device: {device}")

        splits = torch.load(tensors_path, map_location="cpu", weights_only=True)
        seq_len = preprocess_manifest["preprocess_params"]["seq_len"]
        # +1 for the mask/pad sentinel on top of the real quantized port classes
        # (port_top_k explicit ports + 3 IANA range buckets, see schema.port_vocab_size).
        num_port_classes = (
            port_vocab_size(preprocess_manifest["preprocess_params"]["port_top_k"]) + 1
        )

        train_cont, train_proto, train_port, train_mask = (
            splits["train"][k] for k in ("cont", "proto", "port", "mask")
        )
        dataloader = DataLoader(
            TensorDataset(train_cont, train_proto, train_port, train_mask),
            batch_size=params.batch_size,
            shuffle=True,
            drop_last=True,
        )
        if len(dataloader) == 0:
            raise ValueError("Training dataloader is empty.")

        eval_cont, eval_proto, eval_port, eval_mask = (
            splits["eval"][k] for k in ("cont", "proto", "port", "mask")
        )
        eval_dataloader = DataLoader(
            TensorDataset(eval_cont, eval_proto, eval_port, eval_mask),
            batch_size=params.batch_size,
            shuffle=False,
        )

        model, new_metrics = train_model(
            dataloader,
            eval_dataloader,
            params,
            seq_len,
            num_port_classes,
            device,
            initial_state_dict=initial_state_dict,
            initial_best_eval_loss=initial_best_eval_loss,
            initial_best_epoch=initial_best_epoch,
            epoch_offset=epoch_offset,
        )
        metrics = {**previous_metrics, **new_metrics}

        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), target_dir / "model.pt")
        print(
            f"Saved weights are from epoch {metrics['best_epoch']} (best eval loss: {metrics['best_eval_loss']:.6f})"
        )

        write_manifest(
            target_dir,
            {
                "dataset_id": dataset_id,
                "version": version,
                "created_at": now_iso(),
                "training_params": params.to_dict(),
                "input": {
                    "dataset_id": preprocess_manifest.get("dataset_id"),
                    "version": preprocess_manifest.get("version"),
                    "preprocessed_dir": str(preprocessed_dir),
                    "preprocess_params": preprocess_manifest.get("preprocess_params"),
                },
                "metrics": metrics,
                "output_file": "model.pt",
            },
        )

        print(f"Trained model saved to: {target_dir / 'model.pt'}")
        return target_dir

    @staticmethod
    def _coerce_params(params):
        if params is None:
            return TrainingParams()
        if isinstance(params, dict):
            return TrainingParams(**params)
        return params

    @staticmethod
    def _resolve_device(choice: str) -> torch.device:
        if choice == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(choice)


def load_trained_model(
    dataset_id: str,
    version: str,
    models_root: Path = TRAINED_MODELS_DIR,
    map_location: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Reconstruct an AnoBert model (architecture + weights) from a saved checkpoint.

    Reads <models_root>/<dataset_id>/<version>/metadata.yaml for the architecture
    and preprocessing seq_len, then loads model.pt's state_dict into it. Returns
    the model (in eval mode) and the manifest, e.g. to inspect manifest["metrics"]
    or to pass the model on for further training via TrainerHandler.run(resume=True).
    """
    target_dir = Path(models_root) / dataset_id / version
    manifest = read_manifest(target_dir)
    training_params = TrainingParams.from_dict(manifest["training_params"])
    seq_len = manifest["input"]["preprocess_params"]["seq_len"]
    num_port_classes = port_vocab_size(manifest["input"]["preprocess_params"]["port_top_k"]) + 1

    model = AnoBert(
        num_continuous_features=training_params.num_cont_features,
        d_model=training_params.d_model,
        nhead=training_params.n_heads,
        num_layers=training_params.num_layers,
        max_seq_len=seq_len,
        num_port_classes=num_port_classes,
    )
    state_dict = torch.load(
        target_dir / manifest["output_file"], map_location=map_location, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, manifest
