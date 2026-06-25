from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from context_based_anomalous_flow_detector.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
"""
Load UNSW-NB15 CSV from data/raw/.../NF-UNSW-NB15-v3_raw.csv.

Define a UNSWFlowDataset(torch.utils.data.Dataset) that returns single flows (and later pairs of flows for contrastive learning).

Train/val split logic can live here or inside models/train.py.

"""
app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
