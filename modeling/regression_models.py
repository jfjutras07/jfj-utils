import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#---Function:linear_regression---
def linear_regression(train_df, test_df, outcome, predictors, for_stacking=False, **model_params):
    """
    Standard Linear Regression (Ordinary Least Squares).
    Handles both simple and multiple regression based on the predictors list.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LinearRegression(**model_params))
    ])
    
    if for_stacking: return base_pipe
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]
    
    base_pipe.fit(X_train, y_train)
    
    y_pred_train = base_pipe.predict(X_train)
    y_pred_test = base_pipe.predict(X_test)
    
    print(f"--- Linear Regression Summary ---")
    print(f"Predictors: {len(predictors)}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print(f"RMSE (Test): {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
    
    # Extracting coefficients for feature importance
    model = base_pipe.named_steps['model']
    coef_df = pd.DataFrame({
        'Feature': predictors,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    print("\nModel Coefficients:")
    print(coef_df.to_string(index=False))
    print("-" * 35)
    
    return base_pipe

#---Function:polynomial_regression---
def polynomial_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    Polynomial regression with automated degree tuning.
    Finds the optimal degree to capture non-linear relationships.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression(**model_params))
    ])
    
    if for_stacking: return base_pipe
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]
    
    param_grid = {'poly__degree': [1, 2, 3]}
    
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    print(f"--- Polynomial Regression Summary ---")
    print(f"Best Degree: {grid_search.best_params_['poly__degree']}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print(f"RMSE (Test): {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
    print("-" * 35)
    
    return best_model

#---Function:quantile_regression---
def quantile_regression(train_df, test_df, outcome, predictors, quantile=0.5, cv=5, for_stacking=False, **model_params):
    """
    Quantile regression with automated alpha tuning.
    Predicts the specific quantile (e.g., 0.5 for median) instead of the mean.
    """
    # Default params merge with user-specified model_params
    params = {'quantile': quantile, 'solver': 'highs'}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', QuantileRegressor(**params))
    ])
    
    if for_stacking: return base_pipe
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__alpha': [0, 0.01, 0.1, 1.0]}
    
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    print(f"--- Quantile Regression (q={quantile}) Summary ---")
    print(f"Best Alpha: {grid_search.best_params_['model__alpha']}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print(f"MAE (Test): {mean_absolute_error(y_test, y_pred_test):.4f}")
    print("-" * 35)
    
    return best_model

#---Function:robust_regression---
def robust_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    Robust regression using Huber loss.
    Reduces the influence of outliers through automated epsilon tuning.
    """
    params = {'max_iter': 1000}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', HuberRegressor(**params))
    ])
    
    if for_stacking: return base_pipe
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]
    
    param_grid = {
        'model__epsilon': [1.35, 1.5, 1.75, 2.0],
        'model__alpha': [0.0001, 0.001, 0.01]
    }
    
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    print(f"--- Robust Regression (Huber) Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print(f"MAE (Test): {mean_absolute_error(y_test, y_pred_test):.4f}")
    print("-" * 35)
    
    return best_model
