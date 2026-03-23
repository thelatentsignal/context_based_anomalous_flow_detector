from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from context_based_anomalous_flow_detector.config import PROCESSED_DATA_DIR
""" Implement a FlowEncoder(nn.Module) that turns raw flow fields into a single embedding vector.

Contain vocab-building helpers (e.g. build_ip_vocab, build_port_vocab) that you can call once before training.

 """
app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
