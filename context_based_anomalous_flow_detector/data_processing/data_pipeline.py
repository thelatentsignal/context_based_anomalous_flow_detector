# data_processing/data_pipeline.py
from hamilton import driver
import pandas as pd

from context_based_anomalous_flow_detector.paths import UNIFLOWS_V1, PROCESSED_DIR
import data_processing.preprocess_data as preprocess_data
from hamilton import base

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    adapter = base.SimplePythonGraphAdapter(base.DictResult())
    dr = driver.Driver({}, preprocess_data, adapter=adapter)
    #dr = driver.Driver({}, preprocess_data)
    results = dr.execute(["uniflows"])
    df_uniflows: pd.DataFrame = results["uniflows"]

    df_uniflows.to_csv(UNIFLOWS_V1, index=False)
    print(f"Saved uniflows to {UNIFLOWS_V1}")

if __name__ == "__main__":
    main()
