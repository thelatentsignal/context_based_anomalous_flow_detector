"""AnomalyDetectionHandler: scores flows by leave-one-out masked reconstruction error.

The model was trained as a masked-token predictor: it only ever received a training signal
for positions that were masked (see train_step in training/train.py), never for positions fed
in unmasked. So scoring anomalies by feeding a flow through unmasked and comparing directly to
the output (as a plain autoencoder would) compares against never-trained, uncalibrated output.

Instead, this mirrors the task the model was actually trained on: for every real (non-padded)
position, mask exactly that position and let the model reconstruct it from its context. The
weighted reconstruction error at that position -- the same weighting used in the training loss
-- is that flow's anomaly score. A flow the model reconstructs poorly, given its surrounding
context, is one the model finds "surprising", i.e. anomalous.

Unlike EvaluationHandler's single random ~15% masking pass, every real position is masked and
scored exactly once (leave-one-out), giving a complete, deterministic score per flow instead of
a noisy estimate over a random subsample.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from context_based_anomalous_flow_detector.paths import BASE_DATA_PATH, TRAINED_MODELS_DIR
from context_based_anomalous_flow_detector.schema import (
    CONT_PAD_VALUE,
    PORT_PAD_VALUE,
    PROTO_PAD_VALUE,
    now_iso,
)
from context_based_anomalous_flow_detector.training.trainer_handler import load_trained_model


@dataclass
class AnomalyDetectionParams:
    """Configuration for leave-one-out anomaly scoring. Weights mirror the training loss."""

    batch_size: int = 64
    device: str = "auto"  # "auto", "cuda", "cpu"
    cont_weight: float = 30.0
    proto_weight: float = 1.0
    port_weight: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "device": self.device,
            "cont_weight": self.cont_weight,
            "proto_weight": self.proto_weight,
            "port_weight": self.port_weight,
        }


class AnomalyDetectionHandler:
    """Loads a trained model and scores every real flow in a split via leave-one-out
    masked reconstruction error, then reports how well that score separates the
    ground-truth benign/attack labels carried alongside the preprocessed tensors.
    """

    def __init__(self, base_path: Path = BASE_DATA_PATH, models_root: Path = TRAINED_MODELS_DIR):
        self.base_path = Path(base_path)
        self.models_root = Path(models_root)

    def run(
        self,
        dataset_id: str,
        version: str,
        split: str = "test",
        params: AnomalyDetectionParams | dict[str, Any] | None = None,
    ) -> Path:
        """Score `split` for dataset_id/version. Returns the model dir now containing
        anomaly_detection.json (summary metrics) and anomaly_scores.pt (raw per-flow
        scores/labels/mask, for downstream plotting or thresholding)."""
        params = self._coerce_params(params)
        model, _ = load_trained_model(dataset_id, version, models_root=self.models_root)

        device = self._resolve_device(params.device)
        model.to(device)
        model.eval()

        preprocessed_dir = self.base_path / dataset_id / version / "preprocessed"
        tensors_path = preprocessed_dir / "tensors.pt"
        if not tensors_path.exists():
            raise FileNotFoundError(f"Preprocessed tensors not found: {tensors_path}.")

        splits = torch.load(tensors_path, map_location="cpu", weights_only=True)
        if split not in splits:
            raise ValueError(f"Split '{split}' not in preprocessed tensors: {list(splits)}")

        data = splits[split]
        cont, proto, port, mask, label = (
            data[k] for k in ("cont", "proto", "port", "mask", "label")
        )

        scores = self._score_leave_one_out(model, cont, proto, port, mask, params, device)

        real = (~mask).numpy()
        real_scores = scores.numpy()[real]
        real_labels = label.numpy()[real]
        metrics = self._score_metrics(real_scores, real_labels)

        model_dir = self.models_root / dataset_id / version
        results = {
            "dataset_id": dataset_id,
            "version": version,
            "split": split,
            "created_at": now_iso(),
            "params": params.to_dict(),
            "metrics": metrics,
            "output_file": "anomaly_detection.json",
        }
        with open(model_dir / "anomaly_detection.json", "w") as f:
            json.dump(results, f, indent=2)

        torch.save({"scores": scores, "labels": label, "mask": mask}, model_dir / "anomaly_scores.pt")

        self._print_summary(results)
        return model_dir

    @staticmethod
    def _score_leave_one_out(
        model: nn.Module,
        cont: torch.Tensor,
        proto: torch.Tensor,
        port: torch.Tensor,
        mask: torch.Tensor,
        params: AnomalyDetectionParams,
        device: torch.device,
    ) -> torch.Tensor:
        """Mask every real position exactly once and score it from its context.

        Returns a [N, seq_len] tensor of per-flow anomaly scores (0 at padded positions).
        """
        n, seq_len, _ = cont.shape
        scores = torch.zeros(n, seq_len)

        dataloader = DataLoader(
            TensorDataset(cont, proto, port, mask), batch_size=params.batch_size, shuffle=False
        )

        offset = 0
        with torch.no_grad():
            for cont_b, proto_b, port_b, mask_b in dataloader:
                b = cont_b.shape[0]
                cont_b = cont_b.to(device)
                proto_b = proto_b.to(device)
                port_b = port_b.to(device)
                mask_b = mask_b.to(device)

                scores[offset : offset + b] = AnomalyDetectionHandler._score_batch(
                    model, cont_b, proto_b, port_b, mask_b, params
                )
                offset += b

        return scores

    @staticmethod
    def _score_batch(
        model: nn.Module,
        cont_b: torch.Tensor,
        proto_b: torch.Tensor,
        port_b: torch.Tensor,
        mask_b: torch.Tensor,
        params: AnomalyDetectionParams,
    ) -> torch.Tensor:
        """Score every position of a batch in one forward pass.

        Each sequence is replicated seq_len times; replica r has only position r masked.
        This is mathematically equivalent to seq_len separate leave-one-out forward passes
        but runs as a single larger batched call.
        """
        device = cont_b.device
        b, seq_len, num_cont = cont_b.shape

        cont_rep = cont_b.unsqueeze(1).expand(b, seq_len, seq_len, num_cont).clone()
        proto_rep = proto_b.unsqueeze(1).expand(b, seq_len, seq_len).clone()
        port_rep = port_b.unsqueeze(1).expand(b, seq_len, seq_len).clone()
        mask_rep = mask_b.unsqueeze(1).expand(b, seq_len, seq_len).reshape(b * seq_len, seq_len)

        pos = torch.arange(seq_len, device=device)
        cont_rep[:, pos, pos, :] = CONT_PAD_VALUE
        proto_rep[:, pos, pos] = PROTO_PAD_VALUE
        port_rep[:, pos, pos] = PORT_PAD_VALUE

        cont_rep = cont_rep.reshape(b * seq_len, seq_len, num_cont)
        proto_rep = proto_rep.reshape(b * seq_len, seq_len)
        port_rep = port_rep.reshape(b * seq_len, seq_len)

        pred_cont, pred_proto, pred_port = model(
            cont_rep, proto_rep, port_rep, key_padding_mask=mask_rep
        )

        pred_cont = pred_cont.reshape(b, seq_len, seq_len, num_cont)
        pred_proto = pred_proto.reshape(b, seq_len, seq_len, -1)
        pred_port = pred_port.reshape(b, seq_len, seq_len, -1)

        # Keep only each replica's own masked position's prediction: [b, seq_len, ...]
        pred_cont_at_pos = pred_cont[:, pos, pos, :]
        pred_proto_at_pos = pred_proto[:, pos, pos, :]
        pred_port_at_pos = pred_port[:, pos, pos, :]

        cont_err = torch.mean((pred_cont_at_pos - cont_b) ** 2, dim=-1)
        proto_err = nn.functional.cross_entropy(
            pred_proto_at_pos.reshape(b * seq_len, -1), proto_b.reshape(b * seq_len), reduction="none"
        ).reshape(b, seq_len)
        port_err = nn.functional.cross_entropy(
            pred_port_at_pos.reshape(b * seq_len, -1), port_b.reshape(b * seq_len), reduction="none"
        ).reshape(b, seq_len)

        scores = (
            params.cont_weight * cont_err + params.proto_weight * proto_err + params.port_weight * port_err
        )
        scores[mask_b] = 0.0
        return scores.cpu()

    @staticmethod
    def _score_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        n_benign = int((labels == 0).sum())
        n_attack = int((labels == 1).sum())
        metrics: dict[str, Any] = {
            "n_flows": len(scores),
            "n_benign": n_benign,
            "n_attack": n_attack,
            "score_mean_benign": float(scores[labels == 0].mean()) if n_benign else None,
            "score_mean_attack": float(scores[labels == 1].mean()) if n_attack else None,
        }
        if n_benign and n_attack:
            metrics["roc_auc"] = float(roc_auc_score(labels, scores))
            metrics["average_precision"] = float(average_precision_score(labels, scores))
        return metrics

    @staticmethod
    def _coerce_params(
        params: AnomalyDetectionParams | dict[str, Any] | None,
    ) -> AnomalyDetectionParams:
        if params is None:
            return AnomalyDetectionParams()
        if isinstance(params, dict):
            return AnomalyDetectionParams(**params)
        return params

    @staticmethod
    def _resolve_device(choice: str) -> torch.device:
        if choice == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(choice)

    @staticmethod
    def _print_summary(results: dict[str, Any]) -> None:
        metrics = results["metrics"]
        print(
            f"\nAnomaly detection on split '{results['split']}' "
            f"({results['dataset_id']}/{results['version']}):"
        )
        print(
            f"  flows scored     : {metrics['n_flows']:,} "
            f"(benign: {metrics['n_benign']:,}, attack: {metrics['n_attack']:,})"
        )
        if metrics["score_mean_benign"] is not None:
            print(f"  mean score benign: {metrics['score_mean_benign']:.4f}")
        if metrics["score_mean_attack"] is not None:
            print(f"  mean score attack: {metrics['score_mean_attack']:.4f}")
        if "roc_auc" in metrics:
            print(f"  ROC AUC          : {metrics['roc_auc']:.4f}")
            print(f"  Average precision: {metrics['average_precision']:.4f}")
