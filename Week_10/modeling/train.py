from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import typer
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DATA_DIR


app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "education_career_features.csv",
):
    # Step 1: Split the data into features and target variable
    df = pd.read_csv(features_path)
    features = ["University_Ranking", "University_GPA", "Internships_Completed", "Projects_Completed", 
                "Soft_Skills_Score", "Networking_Score", "Career_Satisfaction"]
    X = df[features]
    y = df["Starting_Salary"]

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("Data split into training and testing sets.")

    # Step 3: Scaling the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 4: Train multiple models
    logger.info("Training multiple models...")
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print (f"Model: {name}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R^2 Score: {r2:.2f}\n")

    logger.success("Training complete.")

    # Step 5: Show feature importance for each model
    rf = models["RandomForest"]
    feature_importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title("Feature Importance from Random Forest")
    plt.show()

    gb = models["GradientBoosting"]
    feature_importances = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title("Feature Importance from Gradient Boosting")
    plt.show()

    lr = models["LinearRegression"]
    feature_importances = pd.Series(lr.coef_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title("Feature Importance from Linear Regression")
    plt.show()


if __name__ == "__main__":
    app()
