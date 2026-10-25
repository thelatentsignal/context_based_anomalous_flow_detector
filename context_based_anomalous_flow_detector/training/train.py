from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from context_based_anomalous_flow_detector.modeling.anobert import AnoBert
from context_based_anomalous_flow_detector.schema import CONT_PAD_VALUE, PROTO_PAD_VALUE


def train_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: tuple[torch.nn.Module, torch.nn.Module],
) -> float:
    """Run one masked-token-prediction step, ignoring padding positions."""
    optimizer.zero_grad()

    device = next(model.parameters()).device
    cont_batch, proto_batch, port_batch, attn_mask_batch = [b.to(device) for b in batch]
    # Mask/pad sentinel is one past the model's last real port class -- reading it off
    # the model instance (rather than a fixed schema constant) means it can never
    # desync from whatever port vocab size this particular model was built with.
    port_pad_value = model.embed.port_dst_embedding.num_embeddings - 1

    batch_size, seq_len, _ = cont_batch.shape

    target_cont = cont_batch
    target_proto = proto_batch
    target_port = port_batch

    # attention mask: True = padding, False = real data
    real = ~attn_mask_batch

    # mask 15% of the real (non-padded) positions
    mask = torch.rand(batch_size, seq_len, device=device) < 0.15
    mask = mask & real

    if not mask.any():
        return 0.0

    masked_cont_batch = cont_batch.clone()
    masked_proto_batch = proto_batch.clone()
    masked_port_batch = port_batch.clone()

    masked_cont_batch[mask] = CONT_PAD_VALUE
    masked_proto_batch[mask] = PROTO_PAD_VALUE
    masked_port_batch[mask] = port_pad_value

    pred_cont, pred_proto, pred_port = model(
        masked_cont_batch,
        masked_proto_batch,
        masked_port_batch,
        key_padding_mask=attn_mask_batch,
    )

    mse_loss, cross_entropy_loss = criterion

    cont_loss = mse_loss(pred_cont[mask], target_cont[mask])
    proto_loss = cross_entropy_loss(pred_proto[mask], target_proto[mask])
    port_loss = cross_entropy_loss(pred_port[mask], target_port[mask])

    loss = 30.0 * cont_loss + 1.0 * proto_loss + 0.5 * port_loss

    loss.backward()
    optimizer.step()

    return loss.item()


def _resolve_device(config: dict[str, Any]) -> torch.device:
    choice = config.get("training", {}).get("device", "auto")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def eval_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    criterion: tuple[torch.nn.Module, torch.nn.Module],
) -> float:
    """Masked-token-prediction loss on a batch, no gradient computation."""
    device = next(model.parameters()).device
    cont_batch, proto_batch, port_batch, attn_mask_batch = [b.to(device) for b in batch]
    port_pad_value = model.embed.port_dst_embedding.num_embeddings - 1

    batch_size, seq_len, _ = cont_batch.shape
    real = ~attn_mask_batch

    mask = torch.rand(batch_size, seq_len, device=device) < 0.15
    mask = mask & real

    if not mask.any():
        return None  # signal "skip this batch" to the caller, don't fake a 0.0

    masked_cont_batch = cont_batch.clone()
    masked_proto_batch = proto_batch.clone()
    masked_port_batch = port_batch.clone()
    masked_cont_batch[mask] = CONT_PAD_VALUE
    masked_proto_batch[mask] = PROTO_PAD_VALUE
    masked_port_batch[mask] = port_pad_value

    pred_cont, pred_proto, pred_port = model(
        masked_cont_batch,
        masked_proto_batch,
        masked_port_batch,
        key_padding_mask=attn_mask_batch,
    )

    mse_loss, cross_entropy_loss = criterion
    cont_loss = mse_loss(pred_cont[mask], cont_batch[mask])
    proto_loss = cross_entropy_loss(pred_proto[mask], proto_batch[mask])
    port_loss = cross_entropy_loss(pred_port[mask], port_batch[mask])

    loss = 30.0 * cont_loss + 1.0 * proto_loss + 0.5 * port_loss
    return loss.item()


def train_model(
    dataloader: DataLoader,
    eval_dataloader: DataLoader,
    params: "TrainingParams",
    seq_len: int,
    num_port_classes: int,
    device: torch.device,
    patience: int = 5,
    min_delta: float = 1e-4,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    initial_best_eval_loss: float = float("inf"),
    initial_best_epoch: int = 0,
    epoch_offset: int = 0,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train an AnoBert model. No file I/O — caller handles persistence.

    `num_port_classes` sizes the port embedding/output head to match whatever the
    preprocessing stage's port quantization produced (see schema.port_vocab_size);
    it's passed in explicitly (like `seq_len`) rather than read off `params`, since
    it's a property of the preprocessed data, not a training hyperparameter.

    Pass `initial_state_dict` (with its `initial_best_eval_loss`/`initial_best_epoch`)
    to warm-start from a previously trained checkpoint instead of a random init.
    `epoch_offset` numbers the metrics keys so a resumed run's epochs don't collide
    with the epochs already recorded for that checkpoint. The returned model's
    weights are always the best-eval-loss weights seen across the checkpoint and
    this call combined -- never just the last epoch run -- so a caller that always
    saves/loads `model.state_dict()` can never regress to a worse checkpoint.
    """
    model = AnoBert(
        num_continuous_features=params.num_cont_features,
        d_model=params.d_model,
        nhead=params.n_heads,
        num_layers=params.num_layers,
        max_seq_len=seq_len,
        num_port_classes=num_port_classes,
    ).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )
    criterion = (nn.MSELoss(), nn.CrossEntropyLoss())

    model.train()
    metrics: dict[str, Any] = {}
    best_eval_loss = initial_best_eval_loss
    best_epoch = initial_best_epoch
    best_state_dict = (
        {k: v.clone() for k, v in initial_state_dict.items()}
        if initial_state_dict is not None
        else None
    )
    epochs_without_improvement = 0
    stopped_epoch = epoch_offset + params.num_epochs

    for epoch in range(params.num_epochs):
        epoch_num = epoch_offset + epoch + 1
        total_loss = 0.0
        for batch in dataloader:
            loss_val = train_step(model, batch, optimizer, criterion)
            total_loss += loss_val
        mean_loss = total_loss / len(dataloader)

        model.eval()
        eval_losses = []
        with torch.no_grad():
            for eval_batch in eval_dataloader:
                loss_val = eval_step(model, eval_batch, criterion)
                if loss_val is not None:
                    eval_losses.append(loss_val)
        mean_eval_loss = sum(eval_losses) / len(eval_losses)
        model.train()

        metrics[f"epoch_{epoch_num}_train_loss"] = mean_loss
        metrics[f"epoch_{epoch_num}_eval_loss"] = mean_eval_loss
        print(
            f"Epoch {epoch_num} (run epoch {epoch + 1}/{params.num_epochs}) — train: {mean_loss:.6f}  eval: {mean_eval_loss:.6f}"
        )

        if mean_eval_loss < best_eval_loss - min_delta:
            best_eval_loss = mean_eval_loss
            best_epoch = epoch_num
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stopped_epoch = epoch_num
                print(
                    f"Early stopping at epoch {stopped_epoch} (best eval: {best_eval_loss:.6f} from epoch {best_epoch})"
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    metrics["best_eval_loss"] = best_eval_loss
    metrics["best_epoch"] = best_epoch
    metrics["stopped_epoch"] = stopped_epoch
    return model, metrics


if __name__ == "__main__":
    train_model()
