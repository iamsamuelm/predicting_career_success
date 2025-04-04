from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from Week_10.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "/Users/samuelbond/Documents/INST414/Semester Project/education_career_success.csv",
    output_path: Path = PROCESSED_DATA_DIR / "/Users/samuelbond/Documents/INST414/Semester Project/education_career_success.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    import pandas as pd

    logger.info(f"Reading data from {input_path}")
    try:
        # Read the CSV file
        df = pd.read_csv(input_path)
        logger.info(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns.")

        # Display the first few rows of the dataset
        logger.info("First few rows of the dataset:")
        logger.info(df.head())

        # Display the data types of each column
        logger.info("Data types of each column:")
        logger.info(df.dtypes)

        # Display the number of missing values in each column
        logger.info("Number of missing values in each column:")
        logger.info(df.isnull().sum())

        # Display the summary statistics of the dataset
        logger.info("Summary statistics of the dataset:")
        logger.info(df.describe())

        # Save the processed data to a new CSV file
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    # -----------------------------------------


if __name__ == "__main__":
    app()
