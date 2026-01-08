import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from IPython.display import display
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#---Function:catboost_regression---
def catboost_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    CatBoost Regressor with GridSearchCV. Compares Train and Test R2.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'iterations': [100, 200],
        'depth': [4, 6],
        'learning_rate': [0.05, 0.1]
    }

    cb = CatBoostRegressor(random_state=42, verbose=0)
    grid_search = GridSearchCV(estimator=cb, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- CatBoost Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)

    return best_model

#---Function:decision_tree_regression---
def decision_tree_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    Decision Tree Regressor with GridSearchCV. Compares Train and Test R2.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_leaf': [1, 5, 10]
    }

    dt = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- Decision Tree Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)

    return best_model

#---Function:knn_regression---
def knn_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    K-Nearest Neighbors Regressor with GridSearchCV. Compares Train and Test R2.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'n_neighbors': [3, 5, 11],
        'weights': ['uniform', 'distance']
    }

    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- KNN Regression Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)

    return best_model

#---Function:lightgbm_regression---
def lightgbm_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    LightGBM Regressor with GridSearchCV. Compares Train and Test R2.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'n_estimators': [100, 500],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    }

    lgb = LGBMRegressor(random_state=42, importance_type='gain', verbosity=-1)
    grid_search = GridSearchCV(estimator=lgb, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    importance_df = pd.DataFrame({
        'Feature': predictors, 
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(5)

    print(f"--- LightGBM Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)

    print("\nFeature Importance (Top 5):")
    display(importance_df)

    return best_model

#---Function:random_forest_regression---
def random_forest_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    Random Forest with GridSearchCV. Compares Train and Test R2 to detect overfitting.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    metrics = {
        "R2_Train": r2_score(y_train, y_pred_train),
        "R2_Test": r2_score(y_test, y_pred_test),
        "MAE_Test": mean_absolute_error(y_test, y_pred_test),
        "RMSE_Test": np.sqrt(mean_squared_error(y_test, y_pred_test))
    }

    importance_df = pd.DataFrame({
        'Feature': predictors,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(5)

    print(f"--- Random Forest Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Train): {metrics['R2_Train']:.4f}")
    print(f"R2 Score (Test): {metrics['R2_Test']:.4f}")
    print(f"Gap (Train-Test): {metrics['R2_Train'] - metrics['R2_Test']:.4f}")
    print(f"MAE (Test): {metrics['MAE_Test']:.4f}")
    print(f"Top 3 Features: {importance_df['Feature'].iloc[:3].tolist()}")
    print("-" * 35)

    print("\nFeature Importance (Top 5):")
    display(importance_df)
    
    return best_model

#---Function:xgboost_regression---
def xgboost_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    XGBoost Regressor with GridSearchCV. Compares Train and Test R2 to detect overfitting.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    metrics = {
        "R2_Train": r2_score(y_train, y_pred_train),
        "R2_Test": r2_score(y_test, y_pred_test),
        "MAE_Test": mean_absolute_error(y_test, y_pred_test),
        "RMSE_Test": np.sqrt(mean_squared_error(y_test, y_pred_test))
    }

    importance_df = pd.DataFrame({
        'Feature': predictors,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(5)

    print(f"--- XGBoost Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Train): {metrics['R2_Train']:.4f}")
    print(f"R2 Score (Test): {metrics['R2_Test']:.4f}")
    print(f"Gap (Train-Test): {metrics['R2_Train'] - metrics['R2_Test']:.4f}")
    print(f"MAE (Test): {metrics['MAE_Test']:.4f}")
    print(f"Top 3 Features: {importance_df['Feature'].iloc[:3].tolist()}")
    print("-" * 35)

    print("\nFeature Importance (Top 5):")
    display(importance_df)
    
    return best_model
