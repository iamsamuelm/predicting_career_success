from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import dump, load
from torch.utils.data import DataLoader, TensorDataset, random_split
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
    features = ['Academic_Performance', 'Extracurricular_Score', 'Career_Satisfaction', 'Work_Life_Balance', 
                'Field_of_Study_Business', 'Field_of_Study_Computer Science', 'Field_of_Study_Engineering', 'Field_of_Study_Law', 
                'Field_of_Study_Mathematics', 'Field_of_Study_Medicine', 'Entrepreneur']
    
    scaler_target = MinMaxScaler()
    df['Career_Success_Score_Scaled'] = scaler_target.fit_transform(df[['Career_Success_Score']])

    X = df[features]
    y = df['Career_Success_Score_Scaled']

    # Step 3: Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.success("Features scaled. Data split into training and testing sets.")
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

    # Save the model
    dump(lasso, MODELS_DIR / "lasso_regression_model.joblib")
    print("Lasso Regression model saved.")
    
    logger.success("Lasso Regression training completed.")
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

    # Save the model
    dump(best_model, MODELS_DIR / "gradient_boosting_regressor.joblib")
    print("Gradient Boosting Regressor model saved.")
    
    logger.success("Gradient Boosting Regressor training completed.")
    logger.info("Training PyTorch Neural Network...")

    # Step 6: Convertn numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Step 7: Create PyTorch dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    logger.success("PyTorch dataset and dataloader created.")

    # Step 8: Define the neural network model
    class CareerSuccessNN(nn.Module):
        def __init__(self):
            super(CareerSuccessNN, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(X_train_tensor.shape[1], 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.model(x)
        
    # Step 9: Define loss and optimizer
    model = CareerSuccessNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 10: Train the model
    num_epochs = 100
    early_stopping_patience = 10
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            batch_y = batch_y.view(-1, 1) 
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        model.eval()
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                val_outputs = model(batch_X)
                val_loss = criterion(val_outputs, batch_y)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "pytorch_neural_network.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Step 11: Load the best model & evaluate
    model.load_state_dict(torch.load(MODELS_DIR / "career_success_nn_model.pth"))

    model.eval()
    with torch.no_grad():
        y_pred_nn = model(X_test_tensor).numpy().flatten()

    # Step 12: Report Metrics
    print("Neural Network Performance:")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_nn):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_nn):.2f}")
    print(f"R^2 Score: {r2_score(y_test, y_pred_nn):.2f}")

    logger.success("Neural Network training completed.")


if __name__ == "__main__":
    app()
