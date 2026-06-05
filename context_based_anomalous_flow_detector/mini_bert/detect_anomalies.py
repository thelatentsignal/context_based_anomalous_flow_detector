import torch
import torch.nn as nn
from pathlib import Path

# Hier importierst du deine bestehenden Klassen aus dem ersten Skript
from mini_bert import MiniBert, get_training_data_bert, create_sequences


def calculate_flow_anomaly_scores(model, batch):
    model.eval()
    cont_batch, proto_batch, port_batch = batch
    B, S, _ = cont_batch.shape
    device = next(model.parameters()).device

    # Tensors auf die 5090 schieben
    cont_batch = cont_batch.to(device)
    proto_batch = proto_batch.to(device)
    port_batch = port_batch.to(device)

    with torch.no_grad():
        # WICHTIG: Wir füttern die echten, UNMASKIERTEN Daten hinein!
        pred_cont, pred_proto, pred_port = model(cont_batch, proto_batch, port_batch)

    # 1. Kontinuierlicher Fehler pro Flow: [B, S]
    # (Abweichung quadratisch berechnen und über die 13 Features mitteln)
    loss_cont = torch.mean((pred_cont - cont_batch) ** 2, dim=2)

    # 2. Kategorialer Fehler (Cross Entropy) pro Flow elementweise berechnen
    loss_proto = torch.zeros(B, S, device=device)
    loss_port = torch.zeros(B, S, device=device)

    # Da nn.CrossEntropyLoss standardmäßig reduziert, nutzen wir die funktionale Variante
    # mit reduction="none", um den Loss für jeden einzelnen Zeitschritt (S) zu behalten.
    for b in range(B):
        loss_proto[b] = nn.functional.cross_entropy(
            pred_proto[b], proto_batch[b], reduction="none"
        )
        loss_port[b] = nn.functional.cross_entropy(
            pred_port[b], port_batch[b], reduction="none"
        )

    # 3. Der finale Anomaly Score pro Flow mit deiner trainierten Gewichtung
    anomaly_scores = (30.0 * loss_cont) + (1.0 * loss_proto) + (0.5 * loss_port)
    return anomaly_scores.cpu().numpy()  # Zurück auf die CPU für Auswertungen/Plots


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Frische Live-Daten laden
    live_df = get_training_data_bert("pfad_zu_neuen_netzwerk_daten.csv")
    live_sequences = create_sequences(live_df, seq_len=34)

    # 2. Dein trainiertes Artefakt laden
    latest_checkpoint = sorted(Path("./checkpoints").glob("*.pt"))[-1]

    model = MiniBert()  # Frische Instanz
    state_dict = torch.load(latest_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    # 3. Anomalien detektieren
    scores = calculate_flow_anomaly_scores(model, live_sequences)

    # 4. Schwellenwert-Filter (Beispiel-Threshold: alles über 8.0 ist verdächtig)
    THRESHOLD = 8.0
    anomalous_indices = np.argwhere(scores > THRESHOLD)

    for batch_idx, flow_idx in anomalous_indices:
        print(
            f"WARNUNG: Anomalie entdeckt in Batch {batch_idx}, Flow-Schritt {flow_idx}! Score: {scores[batch_idx, flow_idx]:.2f}"
        )
