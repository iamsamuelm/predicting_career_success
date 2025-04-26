from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import sys
import math
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Week_10.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "education_career_success.csv",
):
    base_name = "plot"
    extension = ".png"
    counter = 1
    output_path = FIGURES_DIR / f"{base_name}{extension}"
    
    while output_path.exists():
        output_path = FIGURES_DIR / f"{base_name}_{counter}{extension}"
        counter += 1
    logger.info(f"Saving plot to {output_path}")

    # ---- EDA Storytelling ---- #
    # Read the CSV file
    df = pd.read_csv(input_path)

    # Visualize missing values
    msno.matrix(df)
    plt.title("Missing Values Matrix")
    plt.show()
    
    # Initial correlation heatmap
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    # Find outliers using Boxplots
    numeric_columns = ['Age', 'High_School_GPA', 'SAT_Score', 'University_Ranking',
       'University_GPA', 'Internships_Completed', 'Projects_Completed',
       'Certifications', 'Soft_Skills_Score', 'Networking_Score', 'Job_Offers',
       'Starting_Salary', 'Career_Satisfaction', 'Years_to_Promotion',
       'Work_Life_Balance']
    
    # Create subplots & calculates rows dynamically
    num_cols = 4
    num_rows = int(np.ceil(len(df.columns) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_columns): # Loop through numeric columns and create boxplots
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_xlabel(col)

    for j in range(i + 1, len(axes)): # Remove empty subplots
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

    # Finding relationships using bar plots
    num_cols = 4
    num_rows = int(np.ceil(len(numeric_columns) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_columns): # Loop through numeric columns and create bar plots
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Bar Plot of {col}')       

    for j in range(i + 1, len(axes)): # Remove empty subplots
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

    # Distribution of categorical variables
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        print(col, "-->", df[col].nunique())

    n_features = len(categorical_features)
    ncols = 2
    nrows = math.ceil(n_features / ncols)

    plt.figure(figsize=(10, nrows * 5))
    for i, col in enumerate(categorical_features):
        plt.subplot(nrows, ncols, i + 1)
        sns.countplot(x=df[col], palette='viridis')
        plt.xticks(rotation=45)
        plt.title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    app()
