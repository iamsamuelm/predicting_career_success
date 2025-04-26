from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Week_10.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "education_career_success.csv",
    output_path: Path = PROCESSED_DATA_DIR / "education_career_features.csv",
):
    logger.info(f"Reading data from {input_path}")
    try:
        # Read the CSV file
        df = pd.read_csv(input_path)
        print("Dataset Info:")
        print(df.info())

    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")


  

if __name__ == "__main__":
    app()
