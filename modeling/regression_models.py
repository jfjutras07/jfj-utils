import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#---Function:linear_regression---
def linear_regression(train_df, test_df, outcome, predictors, for_stacking=False):
    """
    Standard Ordinary Least Squares (OLS) regression.
    Wrapped in a pipeline with scaling for numerical stability.
    """
    model = LinearRegression()
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    if for_stacking:
        return pipeline
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"--- Linear Regression Summary ---")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print("-" * 35)
    
    return pipeline

#---Function:polynomial_regression---
def polynomial_regression(train_df, test_df, outcome, predictors, degree=2, for_stacking=False):
    """
    Polynomial regression. 
    Adds interaction terms and squared/cubic features before linear regression.
    """
    model_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    if for_stacking:
        return model_pipe
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]
    
    model_pipe.fit(X_train, y_train)
    y_pred = model_pipe.predict(X_test)
    
    print(f"--- Polynomial Regression (Degree {degree}) Summary ---")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print("-" * 35)
    
    return model_pipe

#---Function:quantile_regression---
def quantile_regression(train_df, test_df, outcome, predictors, quantile=0.5, for_stacking=False):
    """
    Quantile regression (Median regression by default).
    Useful for predicting specific percentiles or handling heteroscedasticity.
    """
    model = QuantileRegressor(quantile=quantile, alpha=0, solver='highs')
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    if for_stacking:
        return pipeline
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"--- Quantile Regression (q={quantile}) Summary ---")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print("-" * 35)
    
    return pipeline

#---Function:robust_regression---
def robust_regression(train_df, test_df, outcome, predictors, for_stacking=False):
    """
    Robust regression using Huber loss.
    Less sensitive to outliers than standard OLS.
    """
    model = HuberRegressor(max_iter=1000)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    if for_stacking:
        return pipeline
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"--- Robust Regression (Huber) Summary ---")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print("-" * 35)
    
    return pipeline
