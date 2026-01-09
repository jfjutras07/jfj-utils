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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#---Function:catboost_regression---
def catboost_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    CatBoost Regressor with GridSearchCV. Compares Train and Test R2.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', CatBoostRegressor(random_state=42, verbose=0))
    ])

    if for_stacking:
        return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'model__iterations': [100, 200],
        'model__depth': [4, 6],
        'model__learning_rate': [0.05, 0.1]
    }

    grid_search = GridSearchCV(estimator=base_pipe, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
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

#---Function:compare_tree_models---
def compare_tree_models(train_df, test_df, outcome, predictors, cv=5):
    """
    Executes and compares all tree-based and non-parametric models.
    Displays logs, summary table, and feature importance for the champion.
    """
    print("Starting Tree Models Comparison...")
    print(f"Predictors: {len(predictors)} | CV Folds: {cv}")
    print("-" * 35)

    results = {}
    
    print("\n[1/6] Running CatBoost...")
    results['CatBoost'] = catboost_regression(train_df, test_df, outcome, predictors, cv=cv)
    
    print("\n[2/6] Running Decision Tree...")
    results['DecisionTree'] = decision_tree_regression(train_df, test_df, outcome, predictors, cv=cv)
    
    print("\n[3/6] Running KNN...")
    results['KNN'] = knn_regression(train_df, test_df, outcome, predictors, cv=cv)
    
    print("\n[4/6] Running LightGBM...")
    results['LightGBM'] = lightgbm_regression(train_df, test_df, outcome, predictors, cv=cv)
    
    print("\n[5/6] Running Random Forest...")
    results['RandomForest'] = random_forest_regression(train_df, test_df, outcome, predictors, cv=cv)
    
    print("\n[6/6] Running XGBoost...")
    results['XGBoost'] = xgboost_regression(train_df, test_df, outcome, predictors, cv=cv)

    X_test, y_test = test_df[predictors], test_df[outcome]
    comparison_list = []

    for name, model in results.items():
        y_pred = model.predict(X_test)
        comparison_list.append({
            "Model": name,
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        })

    comparison_df = pd.DataFrame(comparison_list).set_index("Model").sort_values(by="R2", ascending=False)

    print("\n--- Final Tree Models Comparison ---")
    print(comparison_df.to_string())
    print("-" * 35)

    winner_name = comparison_df.index[0]
    winner_model = results[winner_name]

    print(f"\nModel Champion: {winner_name}")
    
    # Check if the winner is a Pipeline or a raw model to extract importance
    actual_model = winner_model.named_steps['model'] if isinstance(winner_model, Pipeline) else winner_model

    if hasattr(actual_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': predictors,
            'Importance': actual_model.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(5)
        
        print(f"Top 5 Feature Importances for {winner_name}:")
        display(importance_df)
    else:
        print(f"Feature importance not available for {winner_name} (e.g., KNN).")

    return winner_model

#---Function:decision_tree_regression---
def decision_tree_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Decision Tree Regressor with GridSearchCV. Compares Train and Test R2.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', DecisionTreeRegressor(random_state=42))
    ])

    if for_stacking:
        return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'model__max_depth': [None, 5, 10, 20],
        'model__min_samples_leaf': [1, 5, 10]
    }

    grid_search = GridSearchCV(estimator=base_pipe, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
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
def knn_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    K-Nearest Neighbors Regressor with GridSearchCV. Compares Train and Test R2.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', KNeighborsRegressor())
    ])

    if for_stacking:
        return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'model__n_neighbors': [3, 5, 11],
        'model__weights': ['uniform', 'distance']
    }

    grid_search = GridSearchCV(estimator=base_pipe, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
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
def lightgbm_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    LightGBM Regressor with GridSearchCV. Compares Train and Test R2.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', LGBMRegressor(random_state=42, importance_type='gain', verbosity=-1))
    ])

    if for_stacking:
        return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'model__n_estimators': [100, 500],
        'model__learning_rate': [0.01, 0.1],
        'model__num_leaves': [31, 50]
    }

    grid_search = GridSearchCV(estimator=base_pipe, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
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
def random_forest_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Random Forest with GridSearchCV. Compares Train and Test R2.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(random_state=42))
    ])

    if for_stacking:
        return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 5, 10, 15],
        'model__min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=base_pipe, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
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
def xgboost_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    XGBoost Regressor with GridSearchCV. Compares Train and Test R2.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBRegressor(random_state=42, objective='reg:squarederror'))
    ])

    if for_stacking:
        return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(estimator=base_pipe, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
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
