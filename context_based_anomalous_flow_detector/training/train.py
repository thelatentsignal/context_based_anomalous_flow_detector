
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from context_based_anomalous_flow_detector.config import load_config, get_dataset_params
from context_based_anomalous_flow_detector.modeling.mini_bert import MiniBert
from context_based_anomalous_flow_detector.persistence import create_run_dir, save_training_run


def train_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: tuple[torch.nn.Module, torch.nn.Module],
) -> float:
    optimizer.zero_grad()

    device = next(model.parameters()).device
    cont_batch, proto_batch, port_batch = [b.to(device) for b in batch]

    batch_size, seq_len, _ = cont_batch.shape

    target_cont = cont_batch
    target_proto = proto_batch
    target_port = port_batch

    masked_cont_batch = cont_batch.clone()
    masked_proto_batch = proto_batch.clone()
    masked_port_batch = port_batch.clone()

    mask = torch.rand(batch_size, seq_len, device=device) < 0.15

    masked_cont_batch[mask] = 0.0
    masked_proto_batch[mask] = 256
    masked_port_batch[mask] = 65536

    pred_cont, pred_proto, pred_port = model(
        masked_cont_batch,
        masked_proto_batch,
        masked_port_batch,
    )

    mse_loss, cross_entropy_loss = criterion

    cont_loss = mse_loss(pred_cont[mask], target_cont[mask])
    proto_loss = cross_entropy_loss(pred_proto[mask], target_proto[mask])
    port_loss = cross_entropy_loss(pred_port[mask], target_port[mask])

    loss = 30.0 * cont_loss + 1.0 * proto_loss + 0.5 * port_loss

    loss.backward()
    optimizer.step()

    return loss.item()


def train_model() -> Path:
    config = load_config()
    dataset_params = get_dataset_params("unsw_nb15")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using hardware accelerator device: {device}")

    tensor_path = Path(dataset_params["processed_data_dir"]) / "tensors.pt"

    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor artifact not found: {tensor_path}")

    splits = torch.load(tensor_path, map_location="cpu", weights_only=True)

    train_cont, train_proto, train_port, _train_attention_mask = splits["train"]

    dataset = TensorDataset(train_cont, train_proto, train_port)

    batch_size = config["training"]["batch_size"]
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model_cfg: dict[str, Any] = config["model"]

    model = MiniBert(
        num_continuous_features=model_cfg["num_cont_features"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["n_heads"],
        num_layers=model_cfg["num_layers"],
        max_seq_len=dataset_params["seq_len"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    criterion = (nn.MSELoss(), nn.CrossEntropyLoss())

    num_epochs = config["training"]["num_epochs"]

    model.train()

    metrics: dict[str, Any] = {}

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in dataloader:
            loss_val = train_step(model, batch, optimizer, criterion)
            total_loss += loss_val

        mean_loss = total_loss / len(dataloader)
        metrics[f"epoch_{epoch + 1}_train_loss"] = mean_loss

        print(
            f"Epoch {epoch + 1}/{num_epochs} complete. "
            f"Mean batch loss: {mean_loss:.6f}"
        )

    run_name = config["project"]["experiment_name"]
    run_dir = create_run_dir(run_name)

    save_training_run(
        model=model,
        config=config,
        metrics=metrics,
        run_dir=run_dir,
    )

    print(f"Saved training run to: {run_dir}")

    return run_dir


if __name__ == "__main__":
    train_model()
#
## --- 6. EXECUTION RUNNER ---
#if __name__ == "__main__":
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    print(f"Using hardware accelerator device: {device}")
#    file_path = "/home/irene/DataSSD/Code/context_based_anomalous_flow_detector/data/unsw_nb15/NF-UNSW-NB15-v3_input_to_bert.csv"
#    SEQ_LEN = 34
#    INPUT_DIM = 15
#    D_MODEL = 32
#    NHEAD = 4
#    NUM_LAYERS = 2
#    MAX_SEQ_LEN = 34
#    BATCH_SIZE = 64
#
#    # Execution Fallback logic for sandbox vs local test environments
#    if Path(file_path).exists():
#        print("Loading real NetFlow processing dataframe...")
#        df = get_training_data_bert(file_path)
#        bert_input = create_sequences(df, seq_len=SEQ_LEN)
#    else:
#        print(
#            "File target path unavailable. Exiting." #Injecting safe 15-dim structural Dummy Tensors..."
#        )
#        sys.exit()
#        # bert_input = create_dummy_data()
#
#    # print("Final Shape Processing Matrix Target Setup:", bert_input.shape)
#
#    dataset = TensorDataset(*bert_input)
#    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#
#    model = MiniBert(input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN)
#    model = model.to(device)  # <-- Push weights to GPU VRAM
#
#    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#    criterion = (nn.MSELoss(), nn.CrossEntropyLoss())
#
#    EPOCHS = 20
#    model.train()
#    for epoch in range(EPOCHS):
#        total_loss = 0.0
#        for batch_idx, batch in enumerate(dataloader):
#            loss_val = train_step(model, batch, optimizer, criterion)
#            total_loss += loss_val
#        print(
#            f"Epoch {epoch+1}/{EPOCHS} complete. Mean Batch Loss value: {total_loss / len(dataloader):.6f}"
#        )
#
#    saved_model = save_model_safely("./checkpoints", model)
#    # Verzeichnis erstellen, falls es nicht existiert
#
#    # todo use tensorboard
#    # todo use zero padding when preparing the data instead of cutting things off
#    # todo do not use mse only, but use another method for the categorical values
#
#    # 1. Pfad definieren
#    # model_path: Path = Path(saved_model.stem())
#
#    # 2. Modell laden (wir übergeben die Klasse selbst, ohne Klammern!)
#    loaded_model = verify_and_load_model(saved_model, MiniBert)
#    loaded_model = loaded_model.to(device)
#    # 2. Mimicking your real separated data stream typesggvG
#    batch_size, seq_len = 64, 34
#
#    simulated_cont  = torch.randn(batch_size, seq_len, 13).to(device)            # Floats
#    simulated_ports = torch.randint(0, 65535, (batch_size, seq_len)).to(device)  # Longs
#    simulated_proto = torch.randint(0, 255, (batch_size, seq_len)).to(device)    # Longs
#
#    with torch.no_grad():
#        out_cont, out_port, out_proto = loaded_model(simulated_cont, simulated_proto, simulated_ports)
#
#    print("--- Production Model Sanity Check Passed ---")
#    print("Continuous Features Out Shape: ", out_cont.shape)  # Expected: [64, 34, 13]
#    print("Port Logits Out Shape:          ", out_port.shape)  # Expected: [64, 34, 65537]
#    print("Protocol Logits Out Shape:      ", out_proto.shape)  # Expected: [64, 34, 257]
##todo mach eine quantisierung über die Ports anstatt jeden einzelnen vorhersagen zu wollen
## todo: 3. Effizientere Token-Maskierung (Die 80-10-10-Regel)
## Aktuell ersetzt du jeden ausgewählten Flow starr zu 100% mit den Maskierungs-Tokens (0.0, 256, 65536). Das entspricht dem Ur-BERT-Paper, führt aber dazu, dass das Modell bei der echten Inferenz (ohne Masken) eine Diskrepanz sieht.
##
## Moderne BERT-Architekturen nutzen beim Erstellen der Maske folgende Verteilung für die ausgewählten 15%:
##
## 80% der Fälle: Ersetzen mit dem echten [MASK]-Token (wie du es tust).
##
## 10% der Fälle: Ersetzen mit einem völlig zufälligen Port/Protokoll (zwingt das Modell dazu, Kontext kritisch zu hinterfragen, statt der Maske blind zu vertrauen).
##
## 10% der Fälle: Den echten Wert unverändert lassen (verbessert die Repräsentation der realen Datenströme).
##
## Das kannst du in deine train_step einbauen, sobald das Basistraining stabil konvergiert.