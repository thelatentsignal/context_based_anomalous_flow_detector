import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from IPython.lib.pretty import MAX_SEQ_LENGTH
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import sys
from importlib.resources import files
import yaml
from context_based_anomalous_flow_detector.utils import load_config, get_dataset_params

# --- 1. DUMMY DATA GEN (FOR QUICK RUNTIME TESTING) ---
def create_dummy_data() -> torch.Tensor:
    # 100 sequences, 34 flows per sequence, 15 features per flow
    data = torch.randn(100, 34, 15)
    # Ensure categorical columns have valid initialization integers before masking
    data[:, :, 13] = torch.randint(0, 1024, (100, 34)).float()  # Random ports
    data[:, :, 14] = torch.randint(0, 254, (100, 34)).float()  # Random protocols

    # Inject noticeable payload anomaly inside sequence 0, flow 2
    data[0, 2, 0:13] = torch.tensor(
        [1.0, 1.0, 0.5, 0.5, 0.8, 0.9, 0.1, 0, 1, 0, 0, 1, 0]
    )
    data[0, 2, 13] = 443.0  # HTTPS
    data[0, 2, 14] = 6.0  # TCP
    return data


class MiniBertEmbeddingLayer(nn.Module):
    def __init__(
            self,
            num_continuous_features: int = 13,
            num_categorical_features: int = 2,
            feature_dim: int = 51,
            d_model: int = 32,
            max_seq_len: int = 34
    ):
        super(MiniBertEmbeddingLayer, self).__init__()
        self.num_continuous_features = num_continuous_features
        self.num_categorical_features = num_categorical_features
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.position_embeddings = nn.Embedding(max_seq_len, d_model)

        # Independent Continuous Feature Extractors (Projecting 1D inputs to feature_dim space)
        self.numerical_projectors = nn.ModuleList(
            [nn.Linear(1, self.feature_dim, bias=False) for _ in range(self.num_continuous_features)]
        )
        # Discrete Token Embeddings (Allocating unique dictionary rows for masking tokens)
        self.port_dst_embedding = nn.Embedding(
            65537, self.feature_dim
        )  # 0-65535 + 1 for the mask
        self.proto_embedding = nn.Embedding(
            257, self.feature_dim
        )  # 0-255   + 1 for the mask

        # Mathematically derived bottleneck fusion layer bounds
        total_concat_dim = (self.num_continuous_features + self.num_categorical_features) * self.feature_dim
        self.proj_layer = nn.Linear(total_concat_dim, d_model)
        # Explicitly stable initialization parameters for compression scaling boundaries
        nn.init.xavier_uniform_(self.proj_layer.weight)
        if self.proj_layer.bias is not None: # setting biases to 0 initially is good practice to start with symmetric distributions
            self.proj_layer.bias.data.fill_(0)

        # Swapped from legacy ReLU to modern GELU to prevent vanishing gradients
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(
            self,
            x_continuous: torch.Tensor, # expected shape [Batch x Seq_Len x 13 ]
            x_proto: torch.Tensor, # expected shape [Batch x Seq_Len ]
            x_port: torch.Tensor # expected shape [Batch x Seq_Len ]
    ) -> torch.Tensor:

        B, S, F = x_continuous.shape # B=Batch, S=Seq_Len, F=Continuous Features (13)
        device = x_continuous.device

        projected_features = []
        for i in range(F):
            numerical_feat = x_continuous[:, :, [i]] # Shape [B, S, 1]
            projected_slice = self.numerical_projectors[i](numerical_feat)
            projected_features.append(projected_slice)

        projected_features.append(self.port_dst_embedding(x_port))

        projected_features.append(self.proto_embedding(x_proto))

        # 13 numerical + 2 categorical tensors = 15 tensors of shape [B, S, self.feature_dim (51)]
        # Concatenating along dim=2 multiplies the feature dimension by 15, thus fused_features has dim [B, S, 765]
        fused_features = torch.cat(projected_features, dim=2)
        # Compress the 765-dimensional features to down to model's d_model size (32 todo increase later)
        embed_tokens = self.activation(self.proj_layer(fused_features))

        # todo replace by something more sophisticated RoPE or sine/cosine oder so
        # Generate position integers [0, 1, 2, ..., S-1]
        position_ids = torch.arange(S, dtype=torch.long, device=device)
        # Expand across the batch dimension to match the input grid
        position_ids = position_ids.unsqueeze(0).expand(B, S) # Shape: [B, S]
        # Look up the position embeddings
        position_embeds = self.position_embeddings(position_ids) # Shape: [B, S, 32])

        # Sum tokens and positions together for the final transformer sequence input
        return self.dropout(embed_tokens + position_embeds) # Shape: [B, S, 32]


# --- 3. THE BERT CONTAINER ---
class MiniBert(nn.Module):
    def __init__(
        self,
        num_continuous_features: int = 13,
        num_categorical_features: int = 2,
        feature_dim: int = 51,
        input_dim: int = 15,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 34,
    ):
        super().__init__()
        self.num_continuous_features = num_continuous_features
        self.num_categorical_features = num_categorical_features
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # initialize the custom embedding layer using the parameters from constructor
        self.embed = MiniBertEmbeddingLayer(
            num_continuous_features = self.num_continuous_features,
            num_categorical_features = num_categorical_features,
            feature_dim = self.feature_dim,
            d_model = self.d_model,
            max_seq_len = self.max_seq_len
        )

        # standard transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = self.nhead,
            dim_feedforward = self.d_model * 4, # why 4?
            activation = "gelu",
            batch_first = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.num_layers)

        # multi-task outputheads
        self.reconstruction_continuous = nn.Linear(self.d_model, self.num_continuous_features) # Output size: 13
        self.reconstruction_port  = nn.Linear(self.d_model, 65537) # Logits for 65536 ports + Mask
        self.reconstruction_proto = nn.Linear(self.d_model, 257) # Logits for 256 protocols + Mask

    def forward(
            self,
            x_continuous: torch.Tensor, # shape [B, S, self.num_continous_features (13)],
            x_proto: torch.Tensor,  # shape [B, S],
            x_port: torch.Tensor # shape [B, S],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # B = Batch Size, S = Sequence Length (34), C = Continuous Features (13)
        B, S, num_continuous = x_continuous.shape
        if num_continuous != self.num_continuous_features:
            # todo throw a shape mismatch error
            print('ERROR! the expected number of continuous features does not match the given input data, this will crash!')
        # 1. Fuse heterogeneous streams into abstract transformer sequence tokens
        x = self.embed(x_continuous, x_proto, x_port)

        # 2. Extract contextual multi-flow representations via deep attention
        context_vectors = self.transformer(x)           # Output Shape: [B, S, 32]

        # 3. Pure mathematical mapping out to specialized target distributions
        out_continuous = self.reconstruction_continuous(context_vectors)  # Shape: [B, S, 13]
        out_port = self.reconstruction_port(context_vectors)              # Shape: [B, S, 65537]
        out_proto = self.reconstruction_proto(context_vectors)            # Shape: [B, S, 257]

        return out_continuous, out_proto, out_port



# --- 4. STEP-WISE TRAINING FUNCTION ---
def train_step(
        model:torch.nn.Module,
        batch:tuple[torch.Tensor, torch.Tensor, torch.Tensor], # shape [batch_size, seq_len, 13], [batch_size, seq_len], [batch_size, seq_len]
        optimizer:torch.optim.Optimizer,
        criterion:torch.nn.Module
)->float:

    optimizer.zero_grad()
    device = next(model.parameters()).device

    cont_batch, proto_batch, port_batch = [b.to(device) for b in batch]
    B, S, _ = cont_batch.shape

    target_cont = cont_batch
    target_proto = proto_batch
    target_port = port_batch

    masked_cont_batch = cont_batch.clone()
    masked_proto_batch = proto_batch.clone()
    masked_port_batch = port_batch.clone()

    # num_continuous_features: int = 13,
    # num_categorical_features: int = 2,
    mask = torch.rand(B, S, device=device) < .15
    masked_cont_batch[mask] = 0.0
    masked_port_batch[mask] = 65536  # Port Mask Slot
    masked_proto_batch[mask] = 256  # Protocol Mask Slot

    prediction_cont, prediction_proto, prediction_port = model(
        masked_cont_batch,   masked_proto_batch, masked_port_batch
    ) # shape three vectors

    # Compute loss strictly evaluated across the targeted masking index
    mse_loss, cross_entropy_loss = criterion
    cont_loss = mse_loss(prediction_cont[mask], target_cont[mask])
    # nicht vergessen! prediction_proto[mask] liefert [N, 257]. target_proto[mask] liefert [N]., evtl. muss ich hier die
    # shapes anpassen
    proto_loss = cross_entropy_loss(prediction_proto[mask], target_proto[mask]) # shape proto_batch [batch_size, seq_length]
    port_loss = cross_entropy_loss(prediction_port[mask], target_port[mask])

    # weigh the individual errors such that they are similar, e.g mse will be E[(X-Y)^2] = 0.166
    # error proto = -log(1/256) = 5.55
    # error port = -log(1/65537) = 11.09
    weight_num = 30 # because .166*30 = 4.98
    weight_proto = 1 # because 1*5.55 = 5.55
    weight_port = .5 # because 11.09/2 ~= 5.5
    loss = weight_num * cont_loss + weight_proto * proto_loss + weight_port*port_loss
    loss.backward()
    optimizer.step()
    return loss.item()



def create_sequences(data: pd.DataFrame, seq_len: int = 34) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits the dataframe into uniform blocks, right now whatever doesnt fit into a whole seq_len is cut off
    (todo: add masking, to make best use of the available training data)
    Cuts the uniform blocks into three tensors, one for the numerical features, and one per categorical (proto and port)

    Returns:
        - cont_seqs:         torch.Tensor [num_sequences, seq_len, 13] (float32)
        - port_seqs:         torch.Tensor [num_sequences x seq_len x 1] (long)
        - proto_seqs:        torch.Tensor [num_sequences x seq_len x 1] (long)
    """
    num_sequences = data.shape[0]//seq_len
    truncated_length = data.shape[0] - num_sequences*seq_len
    num_continuous = 13

    # Create uniform blocks for the three return tensors, start with the numerical values
    cont_cols = data.columns[0:num_continuous]
    cont_np = data[cont_cols].iloc[:num_sequences * seq_len].values.astype(np.float32)
    cont_seqs = np.reshape(cont_np, (num_sequences, seq_len, num_continuous))

    # Uniform block for ports
    port_np = data['portdst'].iloc[:num_sequences * seq_len].values.astype(np.int64)
    port_seqs = np.reshape(port_np, (num_sequences, seq_len))

    # Uniform block for proto
    proto_np = data['proto'].iloc[:num_sequences * seq_len].values.astype(np.int64)
    proto_seqs = np.reshape(proto_np, (num_sequences, seq_len))

    cont_seqs = torch.from_numpy(cont_seqs)
    port_seqs = torch.from_numpy(port_seqs)
    proto_seqs = torch.from_numpy(proto_seqs)

    return (cont_seqs, proto_seqs, port_seqs)

def save_model_safely(
        directory: str | Path,
        model: torch.nn.Module,
        base_name: str='model')->Path:
    # Verzeichnis erstellen, falls es nicht existiert


    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest: Path = Path(directory) / f"{base_name}_{timestamp}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dest)
    return dest

def verify_and_load_model(
    filepath: str | Path,
    model_class: torch.nn.Module
) -> torch.nn.Module:
    """Lädt ein gespeichertes state_dict in eine frische Modell-Instanz
    und versetzt es in den Evaluierungsmodus.
    """
    path: Path = Path(filepath)

    # 1. Prüfen, ob die Datei überhaupt existiert
    if not path.exists():
        raise FileNotFoundError(f"Keine Modelldatei unter {path} gefunden.")

    # 2. Eine leere Instanz der Modell-Architektur erstellen
    # (model_class() ruft den Konstruktor deiner Klasse auf, z.B. MyNetwork())
    loaded_model: torch.nn.Module = model_class()

    # 3. Gewichte laden und in die Architektur einfügen
    # weights_only=True ist ein wichtiger Sicherheitsstandard seit PyTorch 2.4
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    loaded_model.load_state_dict(state_dict)

    # 4. WICHTIG: Modell in den Evaluierungsmodus versetzen
    # Das deaktiviert Dropout und Batch-Normalization für die Inferenz
    loaded_model.eval()

    print(f"Modell erfolgreich von {path.name} geladen und validiert.")
    return loaded_model

# --- 6. EXECUTION RUNNER ---
if __name__ == "__main__":
    # data_dir = Path(params["processed_data_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using hardware accelerator device: {device}")

    # get the config params from the config path, just put root of project as configured in pyproject.toml
    #config_path = files("context_based_anomalous_flow_detector") / "config.yaml"
    #with open(config_path, "r") as file:
    #   config = yaml.safe_load(file)
    #dataset_id = "unsw_nb15"

    # get the config and read all params from the according section:
    config = load_config()
    # 2. Extract specific hyperparameters from their respective YAML parent blocks

    #file_path = "context_based_anomalous_flow_detector/data/unsw_nb15/NF-UNSW-NB15-v3_input_to_bert.csv"
    SEQ_LEN = config["data"]["seq_len"]  # Lives under 'data'
    D_MODEL = config["model"]["d_model"]  # Lives under 'model'
    NHEAD = config["model"]["nhead"]  # Lives under 'model'
    NUM_LAYERS = config["model"]["num_layers"]  # Lives under 'model'

    # Note: Add these keys to your config.yaml under 'model' or 'data' if not already present:
    BATCH_SIZE = config["model"].get("batch_size", 64)  # Fallback to 64 if missing
    EPOCHS = config["model"].get("epochs", 20)  # Fallback to 20 if missing
    LR = config["model"].get("lr", 0.001)  # Fallback to 20 if missing

    # Your continuous features (13) + categorical inputs (2) = 15 total features
    INPUT_DIM = 15


    # Execution Fallback logic for sandbox vs local test environments
    if Path(file_path).exists():
        print("Loading real NetFlow processing dataframe...")
        df = get_training_data_bert(file_path)
        bert_input = create_sequences(df, seq_len=config[dataset_id]["seq_len"])
    else:
        print(
            "File target path unavailable. Exiting." #Injecting safe 15-dim structural Dummy Tensors..."
        )
        sys.exit()
        # bert_input = create_dummy_data()

    # print("Final Shape Processing Matrix Target Setup:", bert_input.shape)

    dataset = TensorDataset(*bert_input)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = MiniBert(input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD,
                      num_layers=NUM_LAYERS,  max_seq_len=MAX_SEQ_LENGTH)
    model = model.to(device)  # <-- Push weights to GPU VRAM

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = (nn.MSELoss(), nn.CrossEntropyLoss())

    #EPOCHS = 20
    model.train()
    for epoch in range(config[dataset_id]['epochs']):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            loss_val = train_step(model, batch, optimizer, criterion)
            total_loss += loss_val
        print(
            f"Epoch {epoch+1}/{EPOCHS} complete. Mean Batch Loss value: {total_loss / len(dataloader):.6f}"
        )

    saved_model = save_model_safely("./checkpoints", model)
    # Verzeichnis erstellen, falls es nicht existiert

    # todo use tensorboard
    # todo use zero padding when preparing the data instead of cutting things off
    # todo do not use mse only, but use another method for the categorical values

    # 1. Pfad definieren
    # model_path: Path = Path(saved_model.stem())

    # 2. Modell laden (wir übergeben die Klasse selbst, ohne Klammern!)
    loaded_model = verify_and_load_model(saved_model, MiniBert)
    loaded_model = loaded_model.to(device)
    # 2. Mimicking your real separated data stream typesggvG

    simulated_cont  = torch.randn(batch_size, seq_len, 13).to(device)            # Floats
    simulated_ports = torch.randint(0, 65535, (batch_size, seq_len)).to(device)  # Longs
    simulated_proto = torch.randint(0, 255, (batch_size, seq_len)).to(device)    # Longs

    with torch.no_grad():
        out_cont, out_port, out_proto = loaded_model(simulated_cont, simulated_proto, simulated_ports)

    print("--- Production Model Sanity Check Passed ---")
    print("Continuous Features Out Shape: ", out_cont.shape)  # Expected: [64, 34, 13]
    print("Port Logits Out Shape:          ", out_port.shape)  # Expected: [64, 34, 65537]
    print("Protocol Logits Out Shape:      ", out_proto.shape)  # Expected: [64, 34, 257]
# todo mach eine quantisierung über die Ports anstatt jeden einzelnen vorhersagen zu wollen
# todo: 3. Effizientere Token-Maskierung (Die 80-10-10-Regel)
# Aktuell ersetzt du jeden ausgewählten Flow starr zu 100% mit den Maskierungs-Tokens (0.0, 256, 65536). Das entspricht dem Ur-BERT-Paper, führt aber dazu, dass das Modell bei der echten Inferenz (ohne Masken) eine Diskrepanz sieht.
#
# Moderne BERT-Architekturen nutzen beim Erstellen der Maske folgende Verteilung für die ausgewählten 15%:
#
# 80% der Fälle: Ersetzen mit dem echten [MASK]-Token (wie du es tust).
#
# 10% der Fälle: Ersetzen mit einem völlig zufälligen Port/Protokoll (zwingt das Modell dazu, Kontext kritisch zu hinterfragen, statt der Maske blind zu vertrauen).
#
# 10% der Fälle: Den echten Wert unverändert lassen (verbessert die Repräsentation der realen Datenströme).
#
# Das kannst du in deine train_step einbauen, sobald das Basistraining stabil konvergiert.