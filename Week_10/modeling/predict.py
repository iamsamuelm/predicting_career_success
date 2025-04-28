from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "education_career_features.csv",
):
    logger.info("Starting model training...")
    # Step 1: Load the data
    df = pd.read_csv(features_path)

    # Step 2: Selecting features and target variable
    features = ['Academic_Performance', 'Extracurricular_Score', 'Career_Satisfaction']
    scaler_target = MinMaxScaler()
    df['Career_Success_Score_Scaled'] = scaler_target.fit_transform(df[['Career_Success_Score']])

    X = df[features]
    y = df['Career_Success_Score_Scaled']

    # Step 3: Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Features scaled. Data split into training and testing sets.")
    logger.info("Training Lasso Regression...")

    # Step 4: Train Lasso Regression
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)

    print("Lasso Regression Performance:")
    print(f"Best alpha: {lasso.alpha_:.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_lasso):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lasso):.2f}")
    print(f"R^2 Score: {r2_score(y_test, y_pred_lasso):.2f}")
    
    logger.info("Lasso Regression training completed.")
    logger.info("Training Gradient Boosting Regressor...")

    # Step 5: Train Gradient Boosting Regressor
    params = {
        'n_estimators': [100, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4]
    }

    gbm = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=gbm, param_grid=params, 
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred_gbm = best_model.predict(X_test)

    print("Gradient Boosting Regressor Performance:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_gbm):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_gbm):.2f}")
    print(f"R^2 Score: {r2_score(y_test, y_pred_gbm):.2f}")
    
    logger.info("Gradient Boosting Regressor training completed.")
    logger.success("Model training completed successfully.")

    # Scatter plot: Actual vs Predicted for Gradient Boosting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_gbm, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Career Success Score')
    plt.ylabel('Predicted Score')
    plt.title('Gradient Boosting: Actual vs Predicted')
    plt.show()


if __name__ == "__main__":
    app()
