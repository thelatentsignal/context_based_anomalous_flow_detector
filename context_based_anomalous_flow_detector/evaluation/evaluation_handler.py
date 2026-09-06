"""EvaluationHandler: evaluates a trained MiniBert model on masked-token prediction.

Analogous to TrainerHandler but for the evaluation stage: reads the test split of the
preprocessed tensors, masks random tokens exactly like training does, and measures how
well the model reconstructs the masked tokens.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from context_based_anomalous_flow_detector.data_processing._manifest import read_manifest
from context_based_anomalous_flow_detector.modeling.mini_bert import MiniBert
from context_based_anomalous_flow_detector.paths import BASE_DATA_PATH, TRAINED_MODELS_DIR
from context_based_anomalous_flow_detector.schema import (
    CONT_PAD_VALUE,
    PORT_PAD_VALUE,
    PROTO_PAD_VALUE,
    now_iso,
)


@dataclass
class EvalParams:
    """Configuration that controls how masked-token evaluation is carried out."""

    seed: int = 42
    mask_ratio: float = 0.15
    batch_size: int = 64
    device: str = "auto"  # "auto", "cuda", "cpu"
    top_k: int = 5  # top-k accuracy for the categorical streams
    max_batches: int | None = None  # cap the number of evaluated batches (None = all)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "mask_ratio": self.mask_ratio,
            "batch_size": self.batch_size,
            "device": self.device,
            "top_k": self.top_k,
            "max_batches": self.max_batches,
        }


class EvaluationHandler:
    """Loads a trained model and scores masked-token reconstruction on a held-out split.

    Main metrics are computed only over randomly masked, non-padded positions:
        - continuous stream: MSE and MAE
        - proto/port streams: top-1 and top-k accuracy
    Baselines are reported alongside so the numbers can be judged: chance accuracy,
    majority-class prediction (from the train split), and a trivial zero guess.
    """

    def __init__(self, base_path: Path = BASE_DATA_PATH, models_root: Path = TRAINED_MODELS_DIR):
        self.base_path = Path(base_path)
        self.models_root = Path(models_root)

    def run(
        self,
        dataset_id: str,
        version: str,
        split: str = "test",
        params: EvalParams | dict[str, Any] | None = None,
    ) -> Path:
        """Evaluate the trained model for dataset_id/version on the given split.

        Returns the model directory that now contains ``evaluation.json``.
        """
        params = self._coerce_params(params)

        model_dir = self.models_root / dataset_id / version
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}. Run the training stage first."
            )

        training_manifest = read_manifest(model_dir)

        preprocessed_dir = self.base_path / dataset_id / version / "preprocessed"
        tensors_path = preprocessed_dir / "tensors.pt"
        if not tensors_path.exists():
            raise FileNotFoundError(
                f"Preprocessed tensors not found: {tensors_path}. "
                f"Run the preprocessing stage first."
            )

        torch.manual_seed(params.seed)
        device = self._resolve_device(params.device)
        print(f"Using hardware accelerator device: {device}")

        splits = torch.load(tensors_path, map_location="cpu", weights_only=True)
        if split not in splits:
            raise ValueError(f"Split '{split}' not in preprocessed tensors: {list(splits)}")

        eval_split = splits[split]
        eval_cont, eval_proto, eval_port, eval_mask = (
            eval_split[k] for k in ("cont", "proto", "port", "mask")
        )

        model = self._load_model(model_dir, training_manifest, eval_cont.shape[1], device)

        dataloader = DataLoader(
            TensorDataset(eval_cont, eval_proto, eval_port, eval_mask),
            batch_size=params.batch_size,
            shuffle=False,
        )

        metrics = self._evaluate_masked_tokens(model, dataloader, splits["train"], params)

        results = {
            "dataset_id": dataset_id,
            "version": version,
            "split": split,
            "created_at": now_iso(),
            "eval_params": params.to_dict(),
            "metrics": metrics,
            "input": {
                "model_file": training_manifest.get("output_file"),
                "preprocessed_dir": str(preprocessed_dir),
                "preprocess_params": training_manifest.get("input", {}).get("preprocess_params"),
            },
            "output_file": "evaluation.json",
        }

        results_path = model_dir / "evaluation.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        self._print_summary(results)
        return model_dir

    @staticmethod
    def _load_model(
        model_dir: Path,
        training_manifest: dict[str, Any],
        seq_len: int,
        device: torch.device,
    ) -> MiniBert:
        model_cfg = training_manifest["training_params"]["model"]
        model = MiniBert(
            num_continuous_features=model_cfg["num_cont_features"],
            d_model=model_cfg["d_model"],
            nhead=model_cfg["n_heads"],
            num_layers=model_cfg["num_layers"],
            max_seq_len=seq_len,
        )
        state_dict = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def _evaluate_masked_tokens(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        train_split: dict[str, torch.Tensor],
        params: EvalParams,
    ) -> dict[str, Any]:
        device = next(model.parameters()).device

        # Majority-class baselines, computed over real (non-padded) train tokens.
        train_real = ~train_split["mask"]
        majority_proto = int(torch.mode(train_split["proto"][train_real], dim=0).values.item())
        majority_port = int(torch.mode(train_split["port"][train_real], dim=0).values.item())

        cont_sq_err = 0.0
        cont_abs_err = 0.0
        cont_abs_err_zero = 0.0
        proto_hits = 0
        proto_majority_hits = 0
        port_hits = 0
        port_topk_hits = 0
        port_majority_hits = 0
        masked_positions = 0
        evaluated_batches = 0

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                cont, proto, port, attn_mask = [b.to(device) for b in batch]

                masked_cont, masked_proto, masked_port, mask = self._mask_tokens(
                    cont, proto, port, attn_mask, params.mask_ratio, device
                )

                if not mask.any():
                    continue

                pred_cont, pred_proto, pred_port = model(
                    masked_cont,
                    masked_proto,
                    masked_port,
                    key_padding_mask=attn_mask,
                )

                real_cont = cont[mask]
                diff = pred_cont[mask] - real_cont
                cont_sq_err += float(diff.square().sum())
                cont_abs_err += float(diff.abs().sum())
                cont_abs_err_zero += float(real_cont.abs().sum())

                real_proto = proto[mask]
                real_port = port[mask]

                proto_hits += int((pred_proto[mask].argmax(dim=-1) == real_proto).sum())
                port_hits += int((pred_port[mask].argmax(dim=-1) == real_port).sum())

                port_pred_topk = torch.topk(
                    pred_port[mask], k=min(params.top_k, pred_port[mask].shape[-1]), dim=-1
                ).indices
                port_topk_hits += int(
                    (port_pred_topk == real_port.unsqueeze(-1)).any(dim=-1).sum()
                )

                proto_majority_hits += int((real_proto == majority_proto).sum())
                port_majority_hits += int((real_port == majority_port).sum())

                masked_positions += int(mask.sum())
                evaluated_batches += 1

                if params.max_batches is not None and evaluated_batches >= params.max_batches:
                    break

        if masked_positions == 0:
            raise RuntimeError(
                "No masked positions produced during evaluation. "
                "The split may be empty or fully padded."
            )

        n_cont_values = masked_positions * model.num_continuous_features
        proto_vocab = model.reconstruction_proto.out_features
        port_vocab = model.reconstruction_port.out_features

        return {
            "evaluated_batches": evaluated_batches,
            "masked_positions": masked_positions,
            "cont_mse": cont_sq_err / n_cont_values,
            "cont_mae": cont_abs_err / n_cont_values,
            "proto_top1_acc": proto_hits / masked_positions,
            "port_top1_acc": port_hits / masked_positions,
            f"port_top{params.top_k}_acc": port_topk_hits / masked_positions,
            "baselines": {
                "proto_majority_top1_acc": proto_majority_hits / masked_positions,
                "port_majority_top1_acc": port_majority_hits / masked_positions,
                "proto_chance_top1": 1.0 / proto_vocab,
                "port_chance_top1": 1.0 / port_vocab,
                f"port_chance_top{params.top_k}": min(params.top_k, port_vocab) / port_vocab,
                "cont_naive_zero_mae": cont_abs_err_zero / n_cont_values,
            },
        }

    @staticmethod
    def _mask_tokens(
        cont: torch.Tensor,
        proto: torch.Tensor,
        port: torch.Tensor,
        attn_mask: torch.Tensor,
        mask_ratio: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask a random fraction of real positions, using the same scheme as training."""
        batch_size, seq_len, _ = cont.shape
        mask = torch.rand(batch_size, seq_len, device=device) < mask_ratio
        mask = mask & ~attn_mask

        masked_cont = cont.clone()
        masked_cont[mask] = CONT_PAD_VALUE
        masked_proto = proto.clone()
        masked_proto[mask] = PROTO_PAD_VALUE
        masked_port = port.clone()
        masked_port[mask] = PORT_PAD_VALUE

        return masked_cont, masked_proto, masked_port, mask

    @staticmethod
    def _coerce_params(params: EvalParams | dict[str, Any] | None) -> EvalParams:
        if params is None:
            return EvalParams()
        if isinstance(params, dict):
            return EvalParams(**params)
        return params

    @staticmethod
    def _resolve_device(choice: str) -> torch.device:
        if choice == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(choice)

    @staticmethod
    def _print_summary(results: dict[str, Any]) -> None:
        print(
            f"\nMasked-token evaluation on split '{results['split']}' "
            f"({results['dataset_id']}/{results['version']}):"
        )
        metrics = results["metrics"]
        baseline = metrics["baselines"]
        top_k = results["eval_params"]["top_k"]

        print(f"  evaluated batches : {metrics['evaluated_batches']}")
        print(f"  masked positions  : {metrics['masked_positions']:,}")
        print("  model:")
        print(
            f"    cont MSE/MAE    : {metrics['cont_mse']:.6f} / {metrics['cont_mae']:.6f}"
            f"  (naive-zero MAE: {baseline['cont_naive_zero_mae']:.6f})"
        )
        print(
            f"    proto top-1 acc : {metrics['proto_top1_acc']:.4f}"
            f"  (majority: {baseline['proto_majority_top1_acc']:.4f}, "
            f"chance: {baseline['proto_chance_top1']:.4f})"
        )
        print(
            f"    port top-1 acc  : {metrics['port_top1_acc']:.6f}"
            f"  (majority: {baseline['port_majority_top1_acc']:.6f}, "
            f"chance: {baseline['port_chance_top1']:.6f})"
        )
        print(
            f"    port top-{top_k} acc : "
            f"{metrics[f'port_top{top_k}_acc']:.6f}"
            f"  (chance: {baseline[f'port_chance_top{top_k}']:.6f})"
        )
