import pandas as pd
df = pd.read_csv('../data/raw/f7546561558c07c5_NFV3DATA-A11964_A11964/data/NF-UNSW-NB15-v3_raw.csv')
#print(df.head())
#print(df.tail())
#print(df['Attack'].unique())
#print(df[df['Attack']=="DoS"])

df.columns