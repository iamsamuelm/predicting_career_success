from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
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
    logger.info("Starting feature engineering...")
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

    # Step 3: Place any missing values back
    X = df.copy()
    if 'Work_Life_Balance' not in X.columns:
        X['Work_Life_Balance'] = 0 
    
    # Step 4: One-hot encoding for categorical variables
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    df = pd.get_dummies(df, columns=['Field_of_Study'], drop_first=True)
    df['Entrepreneur'] = df['Entrepreneurship'].apply(lambda x: 1 if x == 'Yes' else 0)

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

    # Step 7: Transformations to reduce outliers in Starting Salary
    df['Log_Starting_Salary'] = np.log1p(df['Starting_Salary'])

    # Find 1st and 99th percentiles
    lower_bound = df['Log_Starting_Salary'].quantile(0.01)
    upper_bound = df['Log_Starting_Salary'].quantile(0.99)

    # Cap the values
    df['Capped_Starting_Salary'] = df['Log_Starting_Salary'].clip(lower=lower_bound, upper=upper_bound)

    # Step 8: Create Academic Performance Score
    df['Academic_Performance'] = (0.4 * df['University_Ranking'] + 
                                   0.6 * df['University_GPA'])
    
    # Step 9: Create Extracurricular Score
    df['Extracurricular_Score'] = (0.5 * df['Internships_Completed'] + 
                                    0.5 * df['Projects_Completed'] + 
                                    0.3 * df['Soft_Skills_Score'] + 
                                    0.2 * df['Networking_Score'])
    
    # Step 10: Create Composite Career Success Score
    df['Career_Success_Score'] = (0.5 * df['Capped_Starting_Salary'] +
                                  0.5 * df['Job_Offers'] + 
                                  0.3 * df['Career_Satisfaction'] + 
                                  0.2 * (1 - df['Years_to_Promotion']))
    
    # Step 11: Drop original columns
    df_refined = df.drop(columns=['University_Ranking', 'University_GPA', 
                                'Internships_Completed', 'Projects_Completed', 
                                'Soft_Skills_Score', 'Networking_Score', 
                                'Starting_Salary', 'Job_Offers', 
                                'Years_to_Promotion'])
        
    # Step 12: Save the processed data
    df_refined.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    logger.success("Feature engineering completed successfully.")


if __name__ == "__main__":
    app()
