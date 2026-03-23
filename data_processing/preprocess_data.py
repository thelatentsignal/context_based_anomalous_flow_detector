# i want to create what looks like unidirectional flows
import pandas as pd
import numpy as np
"""  Keep preprocess_data.py for one-off data cleaning (e.g. filtering columns, converting timestamps to datetime, 
saving a cleaned CSV into data/processed/unsw_processed.csv).
data_exploration.py and the notebook stay for EDA only.

"""



import pandas as pd
from context_based_anomalous_flow_detector.paths import RAW_NF_UNSW

def raw_nf_unsw() -> pd.DataFrame:
    """Load raw NF-UNSW-NB15-v3 NetFlow data."""
    return pd.read_csv(RAW_NF_UNSW)

def uniflows(raw_nf_unsw: pd.DataFrame) -> pd.DataFrame:
    """First simple uniflow extraction with 9 basic fields."""
    print('I am in uniflows and this is raw_nf.unsw')
    print(raw_nf_unsw.head())
    print(raw_nf_unsw.columns)
    cols = ['OUT_BYTES', 'OUT_PKTS']
    #, 'TCP_FLAGS', 'PROTOCOL', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT']
    res_df = raw_nf_unsw[cols]
    print(res_df.head())

    return res_df

#def create_uniflows():
#    cols = ['OUT_BYTES', 'OUT_PKTS', 'TCP_FLAGS', 'PROTOCOL', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT']
#    df = pd.read_csv('../data/raw/f7546561558c07c5_NFV3DATA-A11964_A11964/data/NF-UNSW-NB15-v3.csv', usecols=cols)
#    print(df.head())
#    #save to preproccesed
#    df.to_csv('../data/processed/uniflows.csv', index=False)


##for a start just extract ips
#ips =  df['IPV4_SRC_ADDR'].unique()
#print('before: ', ips[:5])
#np.random.shuffle(ips)
#print('after: ', ips[:5])
#ips = pd.Series(ips)
#
#save to preproccesed
#ips.to_csv('../data/processed/ips.csv', index=False)
