import pandas as pd
import yaml
from pathlib import Path
import torch
import numpy as np
from hamilton import driver, base
from hamilton.function_modifiers import tag
# __file__ ist der absolute Pfad zu dieser .py Datei
# .parent ist der Ordner 'data_processing'
# .parent.parent ist der Root des Pakets
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PACKAGE_ROOT / "context_based_anomalous_flow_detector" / "config.yaml"

# diese Date ist zweigeteilt. Erst die Transformationen und unten kommt der Hamilton driver


################ 1. Basis-Transformationen ########################################
def raw_data(raw_input: str) -> pd.DataFrame:
    """Load raw NF-UNSW-NB15-v3 NetFlow data."""
    return pd.read_csv(raw_input)


def uniflows(raw_data: pd.DataFrame) -> pd.DataFrame:
    """First simple uniflow extraction with 9 basic fields."""
    cols = [
        "OUT_BYTES",
        "IN_BYTES",
        "OUT_PKTS",
        "IN_PKTS",
        "TCP_FLAGS",
        "MAX_TTL",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "L4_SRC_PORT",
        "L4_DST_PORT",
        "FLOW_DURATION_MILLISECONDS",
        "FLOW_START_MILLISECONDS",
        "PROTOCOL",
    ]
    res_df = raw_data[cols]
    res_df.rename(
        columns={
            "OUT_BYTES": "bytes_out",
            "IN_BYTES": "bytes_in",
            "OUT_PKTS": "packets_out",
            "IN_PKTS": "packets_in",
            "MAX_TTL": "ttl",
            "PROTOCOL": "proto",
            "IPV4_SRC_ADDR": "ipsrc",
            "IPV4_DST_ADDR": "ipdst",
            "L4_SRC_PORT": "portsrc",
            "L4_DST_PORT": "portdst",
            "FLOW_DURATION_MILLISECONDS": "duration",
            "FLOW_START_MILLISECONDS": "starttime",
            "TCP_FLAGS": "tcpflags",
        },
        inplace=True,
    )
    pd.set_option("display.max_columns", None)
    print("res_df: ", res_df)
    return res_df

    ################ 2. Feature Engineering & Skalierung ################################
    # --- 5. DATA INGESTION PIPELINE ---
    # def get_training_data_bert(file_path: str | Path) -> pd.DataFrame:

def sorted_flows(uniflows: pd.DataFrame) -> pd.DataFrame:
    """Sortiert chronologisch nach IP und Zeit, um Zeitabstände korrekt zu berechnen."""
    return uniflows.sort_values(by=["ipsrc", "starttime"]).reset_index(drop=True)

def featurized_data(sorted_flows: pd.DataFrame) -> pd.DataFrame:
    data = sorted_flows.copy()

    # compute time differences
    data["time_delta"] = data.groupby("ipsrc")["starttime"].diff().fillna(0)

    # Scaling Transformations
    data["bytes_in"] = np.log1p(data["bytes_in"].astype(np.float32)) / 20.0
    data["bytes_out"] = np.log1p(data["bytes_out"].astype(np.float32)) / 20.0
    data["packets_in"] = np.log1p(data["packets_in"].astype(np.float32)) / 12.0
    data["packets_out"] = np.log1p(data["packets_out"].astype(np.float32)) / 12.0
    data["duration"] = np.log1p(data["duration"].astype(np.float32)) / 12.0
    data["time_delta"] = np.log1p(data["time_delta"].astype(np.float32)) / 15.0
    data["ttl"] = data["ttl"].astype(np.float32) / 255.0

    # Flag Bitmask Parsing
    flags_series = data["tcpflags"].fillna(0).astype(np.int64)
    data["tcp_fin"] = ((flags_series & 1) > 0).astype(np.float32)
    data["tcp_syn"] = ((flags_series & 2) > 0).astype(np.float32)
    data["tcp_rst"] = ((flags_series & 4) > 0).astype(np.float32)
    data["tcp_psh"] = ((flags_series & 8) > 0).astype(np.float32)
    data["tcp_ack"] = ((flags_series & 16) > 0).astype(np.float32)
    data["tcp_urg"] = ((flags_series & 32) > 0).astype(np.float32)

    data["portdst"] = data["portdst"].fillna(0).astype(np.int64)
    data["proto"] = data["proto"].fillna(0).astype(np.int64)

    return data
    #feature_order = [
    #    "bytes_in",
    #    "bytes_out",
    #    "packets_in",
    #    "packets_out",
    #    "ttl",
    #    "duration",
    #    "time_delta",
    #    "tcp_fin",
    #    "tcp_syn",
    #    "tcp_rst",
    #    "tcp_psh",
    #    "tcp_ack",
    #    "tcp_urg",
    #    "portdst",
    #    "proto",
    #]
    #return data[feature_order]

################ 3. Train / Eval / Test Splits & Sequenz-Generierung ################
def _build_tensors(df_slice: pd.DataFrame, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interne Hilfsfunktion, um Dataframe-Abschnitte in 3D/2D Tensors zu schneiden."""
    num_sequences = len(df_slice) // seq_len

    # 13 kontinuierliche Features extrahieren
    cont_cols = ["bytes_in", "bytes_out", "packets_in", "packets_out", "ttl",
                 "duration", "time_delta", "tcp_fin", "tcp_syn", "tcp_rst",
                 "tcp_psh", "tcp_ack", "tcp_urg"]

    cont_np = df_slice[cont_cols].iloc[:num_sequences * seq_len].values.astype(np.float32)
    cont_seqs = torch.from_numpy(cont_np.reshape((num_sequences, seq_len, 13)))

    # Kategoriale Features extrahieren (2D Tensors)
    port_np = df_slice["portdst"].iloc[:num_sequences * seq_len].values.astype(np.int64)
    port_seqs = torch.from_numpy(port_np.reshape((num_sequences, seq_len)))

    proto_np = df_slice["proto"].iloc[:num_sequences * seq_len].values.astype(np.int64)
    proto_seqs = torch.from_numpy(proto_np.reshape((num_sequences, seq_len)))

    return cont_seqs, proto_seqs, port_seqs


def final_tensor_splits(featurized_data: pd.DataFrame, seq_len: int = 34) -> dict[str, tuple]:
    """Führt den chronologischen Split durch und baut die PyTorch-Tensors."""
    total_rows = len(featurized_data)
    train_end = int(total_rows * 0.70)
    eval_end = int(total_rows * 0.85)

    df_train = featurized_data.iloc[:train_end]
    df_eval  = featurized_data.iloc[train_end:eval_end]
    df_test  = featurized_data.iloc[eval_end:]

    return {
        "train": _build_tensors(df_train, seq_len),
        "eval":  _build_tensors(df_eval, seq_len),
        "test":  _build_tensors(df_test, seq_len)
    }

################ 4. Hamilton Execution Driver ########################################

def create_bert_input() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config nicht gefunden unter: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    params = config["unsw_nb15"]
    # Wir fügen den Standard-Wert für seq_len zu den Parametern hinzu
    params["seq_len"] = 34

    # Hamilton nutzt das aktuelle Skript als Modul
    import create_data_unsw_nb15
    dr = driver.Driver(params, create_data_unsw_nb15, adapter=base.DefaultAdapter())

    # Wir fordern gezielt das Endergebnis 'final_tensor_splits' an
    print("Starte Hamilton Daten-Pipeline...")
    outputs = dr.execute(["final_tensor_splits"])
    splits = outputs["final_tensor_splits"]

    # Artefakt sicher auf SSD speichern
    output_path = Path(params["clean_output"]).with_suffix(".pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(splits, output_path)
    print(f"Pipeline erfolgreich! Splits als PyTorch-Artefakt gespeichert unter: {output_path}")

if __name__ == "__main__":
    create_bert_input()
#
#def bert_input_unsw_nb15(uniflows: pd.DataFrame)-> pd.DataFrame:
#    uniflows = uniflows[["ipsrc", "ipdst", "proto", "tcpflags", "portsrc", "portdst","bytes_in", "bytes_out", "packets_in", "packets_out", "ttl", "duration", "starttime"]]
#    # take this data and
#    return uniflows
#
#
#def numerical_values(uniflows: pd.DataFrame) -> pd.DataFrame:
#    uniflows = uniflows[["bytes", "packets", "ttl", "duration"]]
#    # take this data and
#    return uniflows
#
## create_data_unsw_nb15.py
#import pandas as pd
#import torch
#from torch.utils.data import DataLoader, TensorDataset
#from sklearn.preprocessing import StandardScaler
#
#
#def scaled_features(numerical_values: pd.DataFrame) -> pd.DataFrame:
#    """Standardize features so the Linear layer treats them equally."""
#    scaler = StandardScaler()
#    # returns a numpy array
#    scaled_array = scaler.fit_transform(numerical_values)
#    return pd.DataFrame(
#        scaled.array, index=numerical_values.index, columns=numerical_values.columns
#    )
#
################# Hamilton Driver ########################################
#@tag(target='internal')
#def create_bert_input()->None:
#    if not CONFIG_PATH.exists():
#        raise FileNotFoundError(f"Config nicht gefunden unter: {CONFIG_PATH}")
#    # 1. Pfade aus der YAML laden
#    with open(CONFIG_PATH, "r") as f:
#        config = yaml.safe_load(f)
#
#    # Hol dir nur den Teil für diesen speziellen Datensatz
#    params = config["unsw_nb15"]
#    # 2. Hamilton Driver konfigurieren
#    # 'import __main__' referenziert die Funktionen in dieser Datei
#    import __main__
#    dr = driver.Driver(params, __main__, adapter=base.DefaultAdapter())
#    # 3. Ausführen und Speichern
#    outputs = dr.execute(["uniflows","bert_input_unsw_nb15"])
#
#    output_path = Path(params["clean_output"])
#    output_path.parent.mkdir(parents=True, exist_ok=True)
#    outputs["bert_input_unsw_nb15"].to_csv(output_path, index=False)
#
#if __name__ == "__main__":
#    create_bert_input()
#