"""TrainerHandler: reads preprocessed tensors and trains a MiniBert model.

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
from context_based_anomalous_flow_detector.modeling.mini_bert import MiniBert
from context_based_anomalous_flow_detector.paths import BASE_DATA_PATH, TRAINED_MODELS_DIR
from context_based_anomalous_flow_detector.schema import now_iso
from context_based_anomalous_flow_detector.training.train import train_step


@dataclass
class TrainingParams:
    """Configuration that controls how the MiniBert model is trained.

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
                "name": "mini_bert",
                "num_cont_features": self.num_cont_features,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "num_layers": self.num_layers,
            },
        }


class TrainerHandler:
    """Reads preprocessed data and trains a MiniBert model from it.

    Analogue of CanonicalHandler/PreprocessingHandler for the training stage:
    input is the preprocessed tensors of a dataset/version, output is a model
    artifact plus provenance manifest under <models_root>/<dataset_id>/<version>/.
    """

    def __init__(self, base_path: Path = BASE_DATA_PATH, models_root: Path = TRAINED_MODELS_DIR):
        self.base_path = Path(base_path)
        self.models_root = Path(models_root)

    def run(
        self,
        dataset_id: str,
        version: str,
        params: TrainingParams | dict[str, Any] | None = None,
    ) -> Path:
        """Train a MiniBert model on the preprocessed data for dataset_id/version.

        Returns the directory that contains the model artifact and manifest.
        """
        params = self._coerce_params(params)

        preprocessed_dir = self.base_path / dataset_id / version / "preprocessed"
        tensors_path = preprocessed_dir / "tensors.pt"
        if not tensors_path.exists():
            raise FileNotFoundError(
                f"Preprocessed tensors not found: {tensors_path}. "
                f"Run the preprocessing stage first."
            )

        preprocess_manifest = read_manifest(preprocessed_dir)

        torch.manual_seed(params.seed)
        device = self._resolve_device(params.device)
        print(f"Using hardware accelerator device: {device}")

        splits = torch.load(tensors_path, map_location="cpu", weights_only=True)

        train_cont, train_proto, train_port, train_mask = (
            splits["train"][k] for k in ("cont", "proto", "port", "mask")
        )

        seq_len = preprocess_manifest["preprocess_params"]["seq_len"]

        dataset = TensorDataset(train_cont, train_proto, train_port, train_mask)
        dataloader = DataLoader(
            dataset,
            batch_size=params.batch_size,
            shuffle=True,
            drop_last=True,
        )

        if len(dataloader) == 0:
            raise ValueError(
                "Training dataloader is empty. Check that the preprocessed data "
                "contains enough sequences for the configured batch size."
            )

        model = MiniBert(
            num_continuous_features=params.num_cont_features,
            d_model=params.d_model,
            nhead=params.n_heads,
            num_layers=params.num_layers,
            max_seq_len=seq_len,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
        )

        criterion = (nn.MSELoss(), nn.CrossEntropyLoss())

        model.train()

        metrics: dict[str, Any] = {}

        for epoch in range(params.num_epochs):
            total_loss = 0.0

            for batch in dataloader:
                loss_val = train_step(model, batch, optimizer, criterion)
                total_loss += loss_val

            mean_loss = total_loss / len(dataloader)
            metrics[f"epoch_{epoch + 1}_train_loss"] = mean_loss

            print(
                f"Epoch {epoch + 1}/{params.num_epochs} complete. Mean batch loss: {mean_loss:.6f}"
            )

        target_dir = self.models_root / dataset_id / version
        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), target_dir / "model.pt")

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
    def _coerce_params(params: TrainingParams | dict[str, Any] | None) -> TrainingParams:
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
