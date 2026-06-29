import pandas as pd
import os
import yaml
from pathlib import Path
from hamilton import driver, base
from hamilton.function_modifiers import tag
# __file__ ist der absolute Pfad zu dieser .py Datei
# .parent ist der Ordner 'data_processing'
# .parent.parent ist der Root des Pakets
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PACKAGE_ROOT / "context_based_anomalous_flow_detector" / "config.yaml"

# diese Date ist zweigeteilt. Erst die Transformationen und unten kommt der Hamilton driver
################ Transformationen ########################################
def raw_data(raw_input: str) -> pd.DataFrame:
    """Load raw NF-UNSW-NB15-v3 NetFlow data."""
    return pd.read_csv(raw_input)


def uniflows_extended(raw_data: pd.DataFrame) -> pd.DataFrame:
    """First simple uniflow extraction with 9 basic fields."""
    cols = [
        "OUT_BYTES",
        "IN_BYTES",
        "OUT_PKTS",
        "IN_PKTS",
        "TCP_FLAGS",
        "MAX_TTL",
        "PROTOCOL",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "L4_SRC_PORT",
        "L4_DST_PORT",
        "FLOW_DURATION_MILLISECONDS",
        "FLOW_START_MILLISECONDS",
    ]
    res_df = raw_data[cols]
    res_df.rename(
        columns={
            "OUT_BYTES": "bytes_out",
            "IN_BYTES": "bytes_in",
            "OUT_PKTS": "packets_out",
            "IN_PKTS": "packets_in",
            "TCP_FLAGS": "flags",
            "MAX_TTL": "ttl",
            "PROTOCOL": "proto",
            "IPV4_SRC_ADDR": "ipsrc",
            "IPV4_DST_ADDR": "ipdst",
            "L4_SRC_PORT": "portsrc",
            "L4_DST_PORT": "portdst",
            "FLOW_DURATION_MILLISECONDS": "duration",
            "FlOW_START_MILLISECONDS": "starttime",
        },
        inplace=True,
    )
    return res_df





def bert_input_unsw_nb15_extended(uniflows_extended: pd.DataFrame)-> pd.DataFrame:
    uniflows = uniflows_extended[["ipsrc", "ipdst", "in_bytes", "in_packets", "out_bytes", "out_packets","ttl", "duration", "starttime", "portsrc", "portdst"]]
    # take this data and
    return uniflows

def numerical_values(uniflows: pd.DataFrame) -> pd.DataFrame:
    uniflows = uniflows[["bytes", "packets", "ttl", "duration"]]
    # take this data and
    return uniflows

# create_data_unsw_nb15.py
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def scaled_features(numerical_values: pd.DataFrame) -> pd.DataFrame:
    """Standardize features so the Linear layer treats them equally."""
    scaler = StandardScaler()
    # returns a numpy array
    scaled_array = scaler.fit_transform(numerical_values)
    return pd.DataFrame(
        scaled.array, index=numerical_values.index, columns=numerical_values.columns
    )

################ Hamilton Driver ########################################
@tag(target='internal')
def run()->None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config nicht gefunden unter: {CONFIG_PATH}")
    # 1. Pfade aus der YAML laden
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Hol dir nur den Teil für diesen speziellen Datensatz
    params = config["unsw_nb15_extended"]
    # 2. Hamilton Driver konfigurieren
    # 'import __main__' referenziert die Funktionen in dieser Datei
    import __main__
    dr = driver.Driver(params, __main__, adapter=base.DefaultAdapter())
    # 3. Ausführen und Speichern
    outputs = dr.execute(["uniflows_extended","bert_input_unsw_nb15_extended"])

    output_path = Path(params["clean_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outputs["bert_input_unsw_nb15_extended"].to_csv(output_path, index=False)

if __name__ == "__main__":
    run()
