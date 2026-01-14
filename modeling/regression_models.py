import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#---Function:polynomial_regression---
def polynomial_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Polynomial regression with automated degree tuning. 
    English comment: GridSearchCV finds the optimal polynomial degree to balance bias and variance.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    if for_stacking:
        return base_pipe
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]
    
    # English comment: Testing degrees 1, 2, and 3
    param_grid = {'poly__degree': [1, 2, 3]}
    
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print(f"--- Polynomial Regression Optimized ---")
    print(f"Best Degree: {grid_search.best_params_['poly__degree']}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print("-" * 35)
    
    return best_model

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
def quantile_regression(train_df, test_df, outcome, predictors, quantile=0.5, cv=5, for_stacking=False):
    """
    Quantile regression with automated alpha (regularization) tuning.
    English comment: Grid search optimizes the alpha penalty for specific quantile predictions.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', QuantileRegressor(quantile=quantile, solver='highs'))
    ])
    
    if for_stacking:
        return base_pipe
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    # English comment: Testing different regularization strengths
    param_grid = {'model__alpha': [0, 0.01, 0.1, 1.0]}
    
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print(f"--- Quantile Regression (q={quantile}) Optimized ---")
    print(f"Best Alpha: {grid_search.best_params_['model__alpha']}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print("-" * 35)
    
    return best_model

#---Function:robust_regression---
def robust_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Robust regression using Huber loss with automated epsilon tuning.
    English comment: GridSearchCV finds the best threshold for outlier robustness.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', HuberRegressor(max_iter=1000))
    ])
    
    if for_stacking:
        return base_pipe
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]
    
    # English comment: Epsilon controls the sensitivity to outliers
    param_grid = {
        'model__epsilon': [1.1, 1.35, 1.5, 1.75, 2.0],
        'model__alpha': [0.0001, 0.001, 0.01]
    }
    
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print(f"--- Robust Regression (Huber) Optimized ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print("-" * 35)
    
    return best_model
