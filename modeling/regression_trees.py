from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from IPython.display import display

#--- Function : random_forest_regression ---
def random_forest_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    Random Forest with GridSearchCV. Displays Top 5 Features only.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    #Data extraction
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    #Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    #Metrics calculation
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    #Feature importance (Top 5)
    importance_df = pd.DataFrame({
        'Feature': predictors,
        'Importance': best_model.feature_importances_
    })
    
    #Filter non-zero and sort by impact
    active_importance = importance_df[importance_df['Importance'] > 0].copy()
    active_importance = active_importance.sort_values(by='Importance', ascending=False).head(5)

    #Print summary
    print(f"--- Random Forest Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Test): {metrics['R2']:.4f}")
    print(f"MAE (Test): {metrics['MAE']:.4f}")
    print(f"Top 3 Features: {active_importance['Feature'].iloc[:3].tolist()}")
    print("-" * 35)

    print("\nFeature Importance (Sorted by impact):")
    display(active_importance)

#--- Function : xgboost_regression ---
def xgboost_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    XGBoost Regressor with GridSearchCV.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    # Hyperparameter tuning (focused on regularization for small datasets)
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
    y_pred = best_model.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': predictors,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(5)

    print(f"--- XGBoost Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Test): {metrics['R2']:.4f}")
    print(f"MAE (Test): {metrics['MAE']:.4f}")
    print(f"Top 3 Features: {importance_df['Feature'].iloc[:3].tolist()}")
    print("-" * 35)

    print("\nFeature Importance (Sorted by impact):")
    display(importance_df)
