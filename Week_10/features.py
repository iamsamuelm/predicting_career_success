from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from Week_10.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "education_career_success.csv",
    output_path: Path = PROCESSED_DATA_DIR / "education_career_features.csv",
):
    # ---- Feature Engineering ---- #
    # Read the CSV file
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip() # Removing leading/trailing spaces

    # Step 1: Drop unnecessary columns
    df.drop(columns=['Student_ID'], inplace=True)
    print("Column 'Student_ID' has been dropped.")
    
    # Step 2: Handle missing values
    # Fill missing numerical values
    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # Fill missing categorical values
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    # Step 3: Identify the target variable
    X = df.copy()
    if 'Work_Life_Balance' not in X.columns:
        X['Work_Life_Balance'] = 0 
    
    # Step 4: One-hot encoding for categorical variables
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Step 5: Feature scaling
    cols_to_scale = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = MinMaxScaler()
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    print(X.describe())

    # Step 6: Identify statstically significant features if any
    def calculate_vif(data):
        vif_df = pd.DataFrame()
        vif_df["Column"] = data.columns
        vif_df["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        return vif_df
    
    calculate_vif(X[cols_to_scale])
    X.head(5)

    # Step 7: Save the processed data
    X.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    app()
