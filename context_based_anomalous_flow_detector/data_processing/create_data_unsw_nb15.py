import pandas as pd
import yaml
import sys
from pathlib import Path
import torch
import numpy as np
from hamilton import driver, base
# __file__ ist der absolute Pfad zu dieser .py Datei
# .parent ist der Ordner 'data_processing'
# .parent.parent ist der Root des Pakets
#PACKAGE_ROOT = Path(__file__).resolve().parent.parent
#CONFIG_PATH = PACKAGE_ROOT / "context_based_anomalous_flow_detector" / "config.yaml"
#from context_based_anomalous_flow_detector.utils import load_config, get_dataset_params
from context_based_anomalous_flow_detector.config import load_config, get_dataset_params


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
    res_df = raw_data[cols].copy()
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

def sorted_flows(
    uniflows: pd.DataFrame,
    group_cols: list[str],
    time_col: str,
) -> pd.DataFrame:
    return uniflows.sort_values(by=group_cols + [time_col]).reset_index(drop=True)

# todo uses fixed numbers for scaling. Maybe not a good idea. Must change when using other data!
def featurized_data(
        sorted_flows: pd.DataFrame,
        group_cols: list[str],
        time_col:str,
) -> pd.DataFrame:
    data = sorted_flows.copy()

    # compute time differences
    data["time_delta"] = data.groupby(group_cols)[time_col].diff().fillna(0)


    # Scaling Transformations - normalization
    data["bytes_in"] = np.log1p(data["bytes_in"].astype(np.float32)) / 20.0
    data["bytes_out"] = np.log1p(data["bytes_out"].astype(np.float32)) / 20.0
    data["packets_in"] = np.log1p(data["packets_in"].astype(np.float32)) / 12.0
    data["packets_out"] = np.log1p(data["packets_out"].astype(np.float32)) / 12.0
    data["duration"] = np.log1p(data["duration"].astype(np.float32)) / 12.0
    data["time_delta"] = np.log1p(data["time_delta"].astype(np.float32)) / 15.0
    data["ttl"] = data["ttl"].astype(np.float32) / 255.0

    # Flag Bitmask Parsing
    # todo - ist das sinnvoll? die Werte sind fast alle 0
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

################ 3. Train / Eval / Test Splits & Sequenz-Generierung ################
def _build_tensors(
        df_slice: pd.DataFrame,
        group_cols: list[str],
        time_col: str,
        seq_len: int) ->tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        # cont_seqs dim:(n_full, seq_len, num features)
        # proto_seqs dim:(n_full, seq_len, num features)
        # port_seqs dim:(n_full, seq_len, num features)
        # attention_mask dim:(n_full, seq_len, num features)

    """ df_slice is one large pandas dataframe. In this function I subdivide everything into
    packets of seq_len. I do this separately for the continuous features, and the categorical ones (port and proto).

    Every such seq_len will be fed into the bert model. It constitutes one "sentence". Now, I only want "words" from
    ONE IP in one such sentence. But when I cut this dataframe into equal slices of seq_len it may happen that
    one sentence will have mixed IPs, e.g. if my rows have these IPs AAAAAAABBB and my seq_len is 5 I would split this
    into AAAAA AABBB. I do not want AABBB. Thus, what I do in this function is to make sure that AABBB gets
    split into AAXXX, where the XXX is a padding and the BBB get their own packet.
    I also need to inform the model about the padding, thus I also create a mask which is 1 where I pad and 0 else.
    """

    # containers for the groups of size seq_len for the continuous features, proto, port, and mask
    all_cont = []
    all_proto = []
    all_port = []
    all_attention_mask = []


    # 13 kontinuierliche Features extrahieren
    cont_cols = ["bytes_in", "bytes_out", "packets_in", "packets_out", "ttl",
                 "duration", "time_delta", "tcp_fin", "tcp_syn", "tcp_rst",
                 "tcp_psh", "tcp_ack", "tcp_urg"]

    # first I group by the group_cols, initially that was only ip_src, but using a parameter makes this more
    # flexible, I can easily change this to more parameters, e.g. ip_src, ip_dest, in config.yaml.

    # for each group I check if there is a group that is less than seq_len, that one must be masked
    for group_key, group_df in df_slice.groupby(group_cols):
        # does this group have enough flows?
        n = len(group_df)
        n_full = n // seq_len
        full_rows = n_full * seq_len
        left_overs = n % seq_len

        # The easy part. Just create tensors for these because these rows contain only one ipsrc
        # (or whatever group_cols looks for)
        if n_full > 0:
            full_df = group_df.iloc[:full_rows]

            # continuous features
            cont_df = full_df[cont_cols].values.astype(np.float32)
            cont_tensor = torch.from_numpy(
                cont_df.reshape(n_full, seq_len, len(cont_cols))
            )
            all_cont.append(cont_tensor)

            # do this for proto and for port
            # proto
            proto_np = full_df["proto"].values.astype(np.int64)
            proto_tensor = torch.from_numpy(
                    proto_np.reshape(n_full, seq_len)
            )
            all_proto.append(proto_tensor)

            # port
            port_np = full_df["portdst"].values.astype(np.int64)
            port_tensor = torch.from_numpy(
                port_np.reshape(n_full, seq_len)
            )
            all_port.append(port_tensor)

            # create mask for the above
            attention_mask = torch.zeros(
                        (n_full, seq_len),
                        dtype=torch.bool
            )
            all_attention_mask.append(attention_mask)

        # now handle the left overs. They must be padded
        if left_overs > 0:

            leftover_df = group_df.iloc[full_rows:]
            num_pads = seq_len - left_overs

            # as padding values I use values that to not occur in the measured data
            cont_pad = torch.full((seq_len, len(cont_cols)), -1.0, dtype=torch.float32)
            proto_pad = torch.full((seq_len,), 256, dtype=torch.long)
            port_pad = torch.full((seq_len,), 65536, dtype=torch.long)

            # 2. PyTorch Transformer Convention:
            # FALSE = Attend (Real Data) | TRUE = Ignore (Padding)
            attention_mask = torch.full((seq_len,), True, dtype=torch.bool)

            # 3. Extract the real data
            real_cont = torch.from_numpy(leftover_df[cont_cols].values.astype(np.float32))
            real_port = torch.from_numpy(leftover_df["portdst"].values.astype(np.int64))
            real_proto = torch.from_numpy(leftover_df["proto"].values.astype(np.int64))

            # 4. Put the real data on top of the padding to override it
            cont_pad[:left_overs] = real_cont
            proto_pad[:left_overs] = real_proto
            port_pad[:left_overs] = real_port

            # 5. Flip the mask to FALSE for the real data slots so the model reads them
            attention_mask[:left_overs] = False

            all_cont.append(cont_pad.unsqueeze(0))
            all_proto.append(proto_pad.unsqueeze(0))
            all_port.append(port_pad.unsqueeze(0))
            all_attention_mask.append(attention_mask.unsqueeze(0))

    cont_seqs = torch.cat(all_cont, dim=0)
    proto_seqs = torch.cat(all_proto, dim=0)
    port_seqs = torch.cat(all_port, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)

    return cont_seqs, proto_seqs, port_seqs, attention_mask

def save_data_artifacts(splits: dict, config: dict, dataset_id: str, target_dir: Path ) -> None:
    """Handles the physical writing of the tensors and config snapshot to disk."""
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save the PyTorch tensors
    torch.save(splits, target_dir / "tensors.pt")

    # 2. Save the frozen config snapshot for the data section
    data_config = {
        "project": config.get("project", {}),
        "dataset_id": dataset_id,
        "dataset": config["datasets"][dataset_id],
    }

    with open(target_dir / "data_config.yaml", "w") as f:
        yaml.safe_dump(data_config, f, sort_keys=False)

   # with open(target_dir / "data_config.yaml", "w") as f:
   #     yaml.safe_dump(config, f)

   # print(f"Artifacts successfully frozen in: {target_dir}")

#todo do not split
def final_tensor_splits(
        featurized_data: pd.DataFrame,
        group_cols: list[str],     # <-- Hamilton injects this from config
        time_col: str,             # <-- Hamilton injects this from config
        seq_len: int) -> dict[str, tuple]:

    print('featurized data: ', featurized_data.head())
    """Führt den chronologischen Split durch und baut die PyTorch-Tensors."""
    total_rows = len(featurized_data)
    train_end = int(total_rows * 0.70)
    eval_end = int(total_rows * 0.85)

    df_train = featurized_data.iloc[:train_end]
    df_eval  = featurized_data.iloc[train_end:eval_end]
    df_test  = featurized_data.iloc[eval_end:]
    return {
        "train": _build_tensors(df_train, group_cols, time_col, seq_len),
        "eval":  _build_tensors(df_eval, group_cols, time_col, seq_len),
        "test":  _build_tensors(df_test, group_cols, time_col, seq_len)
    }

################ 4. Hamilton Execution Driver ########################################
def create_bert_input(dataset_id:str) -> None:
    params = get_dataset_params(dataset_id)
    config = load_config()

    module = sys.modules[__name__]
    dr = driver.Driver(params, module, adapter=base.DefaultAdapter())

    print("Starte Hamilton Daten-Pipeline...")

    outputs = dr.execute(["final_tensor_splits"])
    splits = outputs["final_tensor_splits"]

    output_dir = Path(params["processed_data_dir"])
    save_data_artifacts(splits, config, dataset_id, output_dir)

    print(f"Pipeline erfolgreich! Splits als PyTorch-Artefakt gespeichert unter: {output_dir}")

if __name__ == "__main__":
    create_bert_input("unsw_nb15")