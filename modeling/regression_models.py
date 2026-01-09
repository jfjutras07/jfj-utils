import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor

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
    pipeline.fit(X_train, y_train)
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
    model_pipe.fit(X_train, y_train)
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
    pipeline.fit(X_train, y_train)
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
    pipeline.fit(X_train, y_train)
    return pipeline
