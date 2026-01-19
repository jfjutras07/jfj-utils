import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from IPython.display import display

#---Function:catboost_regression---
def catboost_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    CatBoost Regressor with GridSearchCV. Optimizes depth and iterations.
    """
    params = {'random_state': 42, 'verbose': 0}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', CatBoostRegressor(**params))
    ])

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'model__iterations': [100, 200],
        'model__depth': [4, 6],
        'model__learning_rate': [0.05, 0.1]
    }

    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
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

#---Function:compare_regression_tree_models---
def compare_regression_tree_models(train_df, test_df, outcome, predictors, cv=5):
    """
    Executes and compares tree-based and non-parametric regression models.
    Returns the champion based on Test R2 score.
    """
    print(f"Starting Tree Models Comparison | Predictors: {len(predictors)}")
    print("-" * 45)

    results = {
        'CatBoost': catboost_regression(train_df, test_df, outcome, predictors, cv),
        'DecisionTree': decision_tree_regression(train_df, test_df, outcome, predictors, cv),
        'KNN': knn_regression(train_df, test_df, outcome, predictors, cv),
        'LightGBM': lightgbm_regression(train_df, test_df, outcome, predictors, cv),
        'RandomForest': random_forest_regression(train_df, test_df, outcome, predictors, cv),
        'XGBoost': xgboost_regression(train_df, test_df, outcome, predictors, cv)
    }

    X_test, y_test = test_df[predictors], test_df[outcome]
    perf_metrics = []

    for name, model in results.items():
        y_pred = model.predict(X_test)
        perf_metrics.append({
            "Model": name,
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        })

    comparison_df = pd.DataFrame(perf_metrics).set_index("Model").sort_values(by="R2", ascending=False)
    
    print("\n--- Final Regression Comparison (Sorted by R2) ---")
    print(comparison_df.to_string())
    
    winner_name = comparison_df.index[0]
    winner_model = results[winner_name]
    print(f"\nCHAMPION: {winner_name}")
    
    actual_model = winner_model.named_steps['model']
    if hasattr(actual_model, 'feature_importances_'):
        feat_imp = pd.DataFrame({
            'Feature': predictors,
            'Importance': actual_model.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(10)
        print("\nTop 10 Feature Importances:")
        display(feat_imp)
    
    return winner_model

#---Function:decision_tree_regression---
def decision_tree_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    Decision Tree Regressor with depth optimization.
    """
    params = {'random_state': 42}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', DecisionTreeRegressor(**params))
    ])

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__max_depth': [None, 5, 10, 20], 'model__min_samples_leaf': [1, 5, 10]}

    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
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
def knn_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    K-Neighbors Regressor. Standardizes features for distance calculation.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', KNeighborsRegressor(**model_params))
    ])

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__n_neighbors': [3, 5, 11], 'model__weights': ['uniform', 'distance']}

    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- KNN Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)

    return best_model

#---Function:lightgbm_regression---
def lightgbm_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    LightGBM Regressor with leaf-wise growth.
    """
    params = {'random_state': 42, 'importance_type': 'gain', 'verbosity': -1}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', LGBMRegressor(**params))
    ])

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__n_estimators': [100, 500], 'model__learning_rate': [0.01, 0.1], 'model__num_leaves': [31, 50]}

    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- LightGBM Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)

    return best_model

#---Function:random_forest_regression---
def random_forest_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    Random Forest Regressor. Reduces variance through bagging.
    """
    params = {'random_state': 42}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(**params))
    ])

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10, 20]}

    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- Random Forest Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)

    return best_model

#---Function:xgboost_regression---
def xgboost_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    XGBoost Regressor. Advanced gradient boosting with regularization.
    """
    params = {'random_state': 42, 'objective': 'reg:squarederror'}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBRegressor(**params))
    ])

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1], 'model__max_depth': [3, 5]}

    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- XGBoost Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
    print("-" * 35)

    return best_model
