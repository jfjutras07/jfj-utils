import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from IPython.display import display
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

#---Function:bayesian_regression---
def bayesian_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Bayesian Ridge Regression with automated tuning.
    English comment: Grid search optimizes the regularization parameters alpha and lambda.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', BayesianRidge())
    ])

    if for_stacking:
        return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    # English comment: Tuning initial hyper-prior parameters
    param_grid = {
        'model__alpha_1': [1e-6, 1e-5],
        'model__lambda_1': [1e-6, 1e-5],
        'model__alpha_init': [None, 1.0]
    }

    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    
    # English comment: Extract predictions and uncertainty
    y_pred, y_std = best_pipeline.named_steps['model'].predict(
        best_pipeline.named_steps['scaler'].transform(
            best_pipeline.named_steps['imputer'].transform(X_test)
        ), 
        return_std=True
    )

    print(f"--- Bayesian Regression Optimized ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred):.4f}")
    print(f"Average Prediction Uncertainty (Std): {np.mean(y_std):.4f}")
    print("-" * 35)

    return best_pipeline

#---Function:gaussian_process_regression---
def gaussian_process_regression(train_df, test_df, outcome, predictors, cv=3, for_stacking=False):
    """
    Gaussian Process Regressor (GPR) with automated noise-level tuning.
    English comment: Optimized for datasets where noise estimation is key.
    """
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(noise_level=1)
    
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42))
    ])

    if for_stacking:
        return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    # English comment: Tuning the alpha parameter (regularization/noise)
    param_grid = {'model__alpha': [1e-10, 1e-5, 1e-2]}
    
    # Note: Reduced CV due to computational cost of GPR
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    
    # English comment: Manual transform for internal predict call to get uncertainty
    X_test_scaled = best_pipeline.named_steps['scaler'].transform(
        best_pipeline.named_steps['imputer'].transform(X_test)
    )
    y_pred, sigma = best_pipeline.named_steps['model'].predict(X_test_scaled, return_std=True)

    print(f"--- Gaussian Process Optimized ---")
    print(f"Best Alpha: {grid_search.best_params_['model__alpha']}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"Average Uncertainty (Sigma): {np.mean(sigma):.4f}")
    print(f"Learned Kernel: {best_pipeline.named_steps['model'].kernel_}")
    print("-" * 35)

    return best_pipeline

#---Function:mlp_regression---
def mlp_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Multi-layer Perceptron regressor with automated grid search.
    Mandatory scaling included in pipeline.
    """
    if for_stacking:
        #Return un-fitted pipeline with default/best hyperparameters for stacking
        model = MLPRegressor(max_iter=1000, random_state=42, early_stopping=True, hidden_layer_sizes=(100, 50))
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'model__hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': [0.0001, 0.01]
    }

    base_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', MLPRegressor(max_iter=1000, random_state=42, early_stopping=True))
    ])

    grid_search = GridSearchCV(estimator=base_pipeline, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    y_pred_test = best_pipeline.predict(X_test)

    print(f"--- Neural Network (MLP) Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Test: {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)

    return best_pipeline

#---Function:svm_regression---
def svm_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Support Vector Regression (SVR) with GridSearchCV and Permutation Importance.
    Mandatory scaling included in pipeline.
    """
    if for_stacking:
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'model__kernel': ['rbf', 'poly'],
        'model__C': [0.1, 1, 10],
        'model__epsilon': [0.01, 0.1]
    }

    base_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', SVR())
    ])

    grid_search = GridSearchCV(estimator=base_pipeline, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    #Permutation Importance for SVR
    perm_importance = permutation_importance(best_pipeline, X_test, y_test, n_repeats=5, random_state=42)
    
    importance_df = pd.DataFrame({
        'Feature': predictors,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False).head(5)

    print(f"--- SVM (SVR) Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred):.4f}")
    print("-" * 35)
    display(importance_df)

    return best_pipeline
