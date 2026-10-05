# Elevate Labs AI & ML Internship - Task 3: Linear Regression

## Overview
This repository contains the implementation of Simple and Multiple Linear Regression models using Python, Pandas, Scikit-learn, and Matplotlib/Seaborn.

## Dataset
The project uses the **California Housing Dataset** fetched directly via `sklearn.datasets.fetch_california_housing`. A local copy is stored under `dataset/housing.csv`.

---

## Solutions to Interview Questions

### 1. What assumptions does linear regression make?
* **Linearity:** The relationship between features and target variable is linear.
* **Independence:** Observations/residuals are independent of each other.
* **Homoscedasticity:** Constant variance of error terms across predictor values.
* **Normality of Residuals:** Residuals follow a normal distribution.
* **No Multicollinearity:** Independent variables are not highly correlated with each other.

### 2. How do you interpret the coefficients?
* **Simple Linear Regression:** The coefficient represents the expected change in the dependent variable ($Y$) for every $1$-unit change in the predictor ($X$).
* **Multiple Linear Regression:** The coefficient represents the change in $Y$ per $1$-unit change in that specific predictor, holding all other features constant.

### 3. What is $R^2$ score and its significance?
* **Definition:** $R^2$ (Coefficient of Determination) measures the proportion of variance in the target variable explained by the model's features.
* **Significance:** $R^2 = 1$ indicates a perfect model, while $R^2 = 0$ means the model performs no better than predicting the target mean. Negative values imply worse performance than the baseline mean model.

### 4. When would you prefer MSE over MAE?
* **MSE (Mean Squared Error):** Penalizes larger errors heavily due to squaring. Preferred when large outliers or deviations are unacceptable.
* **MAE (Mean Absolute Error):** Treats all errors linearly. Preferred when the dataset contains noisy outliers that shouldn't heavily skew the model performance metric.

### 5. How do you detect multicollinearity?
* **Correlation Matrix / Heatmap:** Checking for high correlation coefficients ($r > 0.8$) between feature pairs.
* **Variance Inflation Factor (VIF):** A $VIF > 5 \text{ or } 10$ indicates severe multicollinearity for a feature.

### 6. What is the difference between simple and multiple regression?
* **Simple Linear Regression:** Uses a single predictor variable $X$ to predict $Y$ ($Y = \beta_0 + \beta_1 X$).
* **Multiple Linear Regression:** Uses two or more predictor variables $X_1, X_2, \dots, X_n$ to predict $Y$ ($Y = \beta_0 + \beta_1 X_1 + \dots + \beta_n X_n$).

### 7. Can linear regression be used for classification?
* Standard Linear Regression is **not recommended** for classification because output values are unbounded ($(-\infty, \infty)$) and highly sensitive to outliers.
* **Logistic Regression** (or thresholding linear predictions) is used instead for classification tasks.

### 8. What happens if you violate regression assumptions?
* **Violating Linearity:** Poor model fit; underfitting.
* **Violating Homoscedasticity:** Invalid standard errors, leading to unreliable hypothesis testing ($p$-values and confidence intervals).
* **Violating Multicollinearity:** Unstable coefficients and inflated standard errors.
* **Violating Independence/Normality:** Biased inference and inaccurate confidence intervals.

---

## How to Run

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/ElevateLabs-AIML-Task3-LinearRegression.git
   cd ElevateLabs-AIML-Task3-LinearRegression
