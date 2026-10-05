import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

def load_and_preprocess_data():
    # Load sample dataset (California Housing Dataset)
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    
    # Save a copy locally for reference
    df.to_csv("dataset/housing.csv", index=False)
    print("Dataset loaded and saved to dataset/housing.csv")
    print(f"Dataset Shape: {df.shape}\n")
    
    return df

def run_simple_linear_regression(df):
    print("=" * 50)
    print("1. SIMPLE LINEAR REGRESSION")
    print("=" * 50)
    
    # Using 'MedInc' (Median Income) to predict 'MedHouseVal'
    X = df[['MedInc']]
    y = df['MedHouseVal']
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model initialization and fitting
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Coefficient (Slope): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}\n")
    
    # Plotting Regression Line
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, alpha=0.3, label='Actual Values', color='blue')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Median Income (MedInc)')
    plt.ylabel('Median House Value (MedHouseVal)')
    plt.title('Simple Linear Regression: Income vs House Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('simple_regression_line.png')
    plt.show()

def run_multiple_linear_regression(df):
    print("=" * 50)
    print("2. MULTIPLE LINEAR REGRESSION")
    print("=" * 50)
    
    # Features and Target
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model initialization and fitting
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}\n")
    
    # Coefficient Interpretation
    print("Coefficient Interpretation:")
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    print(coef_df.to_string(index=False))

if __name__ == "__main__":
    import os
    os.makedirs("dataset", exist_ok=True)
    
    dataframe = load_and_preprocess_data()
    run_simple_linear_regression(dataframe)
    run_multiple_linear_regression(dataframe)
