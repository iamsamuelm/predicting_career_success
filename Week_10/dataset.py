from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "/Users/samuelbond/Documents/INST414/Semester Project/education_career_success.csv",
    output_path: Path = PROCESSED_DATA_DIR / "/Users/samuelbond/Documents/INST414/Semester Project/predicting_career_success/data/processed/education_career_success.csv",
):
    logger.info(f"Reading data from {input_path}")
    try:
        # Read the CSV file
        df = pd.read_csv(input_path)
        print("Dataset Info:")
        print(df.info())

        # Display the first few rows of the dataset
        print("First few rows of the dataset:")
        print(df.head())

        # Display the summary statistics of the dataset
        print("Summary statistics of the dataset:")
        print(df.describe())

        # Display the column names of the dataset
        print("Column of the dataset:")
        print(df.columns)

        # Display the number of missing values in each column
        print("Number of missing values in each column:")
        print(df.isnull().sum())

        # Save the DataFrame to a new CSV file
        print(f"Saving the DataFrame to {output_path}")
        df.to_csv(output_path, index=False)
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")


if __name__ == "__main__":
    app()

