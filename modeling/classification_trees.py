import warnings
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from IPython.display import display

# Globally ensure transformers output DataFrames to keep feature names
sklearn.set_config(transform_output="pandas")

#---Function:catboost_classification---
def catboost_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    CatBoost Classifier with GridSearchCV.
    Handles categorical features efficiently using symmetric trees and gradient boosting.
    """
    params = {'random_state': 42, 'verbose': 0, 'auto_class_weights': 'Balanced'}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', CatBoostClassifier(**params))
    ])
    base_pipe.set_output(transform="pandas")

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__iterations': [100, 200], 'model__depth': [4, 6]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- CatBoost Classification Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"F1 Score (Train): {f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"F1 Score (Test): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    print("-" * 35)
    return best_model

# --- Function : compare_classification_tree_models ---
def compare_classification_tree_models(train_df, test_df, outcome, predictors, cv=5):
    """
    Compare tree-based classifiers on given train/test data.
    Feature name warnings for LightGBM/XGBoost are fixed.
    Maintains original print format and pipeline structure.
    """
    print(f"Starting Tree Models Comparison | Predictors: {len(predictors)}")
    print("-" * 45)

    # Train all models (using their default internal settings)
    results = {
        'CatBoost': catboost_classification(train_df, test_df, outcome, predictors, cv),
        'DecisionTree': decision_tree_classification(train_df, test_df, outcome, predictors, cv),
        'LightGBM': lightgbm_classification(train_df, test_df, outcome, predictors, cv),
        'RandomForest': random_forest_classification(train_df, test_df, outcome, predictors, cv),
        'XGBoost': xgboost_classification(train_df, test_df, outcome, predictors, cv)
    }

    # Force X_test to always be a DataFrame with correct columns
    X_test = test_df[predictors] if isinstance(test_df, pd.DataFrame) else pd.DataFrame(test_df, columns=predictors)
    y_test = test_df[outcome] if isinstance(test_df, pd.DataFrame) else pd.Series(test_df[outcome])

    perf_metrics = []

    for name, model in results.items():
        y_pred = model.predict(X_test)

        perf_metrics.append({
            "Model": name,
            "F1_Weighted": f1_score(y_test, y_pred, average='weighted'),
            "Accuracy": accuracy_score(y_test, y_pred)
        })

    # Final comparison
    comparison_df = pd.DataFrame(perf_metrics).set_index("Model").sort_values(by="F1_Weighted", ascending=False)

    print("\n--- Final Tree Comparison (Sorted by F1) ---")
    print(comparison_df.to_string())

    winner_name = comparison_df.index[0]
    winner_model = results[winner_name]

    print(f"\nCHAMPION: {winner_name}")

    # Feature importance extraction
    actual_model = winner_model.named_steps['model']
    feat_imp = pd.DataFrame({
        'Feature': predictors,
        'Importance': actual_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)

    print("\nTop 10 Feature Importances:")
    display(feat_imp)

    return winner_model

#---Function:decision_tree_classification---
def decision_tree_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    Decision Tree Classifier with depth optimization.
    """
    params = {'random_state': 42, 'class_weight': 'balanced'}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', DecisionTreeClassifier(**params))
    ])
    base_pipe.set_output(transform="pandas")

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__max_depth': [None, 5, 10], 'model__min_samples_leaf': [1, 5]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- Decision Tree Classification Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"F1 Score (Train): {f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"F1 Score (Test): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    print("-" * 35)
    return best_model

#---Function:lightgbm_classification---
def lightgbm_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    LightGBM Classifier with leaf-wise growth optimization.
    """
    params = {'random_state': 42, 'verbosity': -1, 'class_weight': 'balanced'}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', LGBMClassifier(**params))
    ])
    base_pipe.set_output(transform="pandas")

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- LightGBM Classification Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"F1 Score (Train): {f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"F1 Score (Test): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    print("-" * 35)
    return best_model

#---Function:random_forest_classification---
def random_forest_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    Random Forest Classifier using Bagging technique.
    """
    params = {'random_state': 42, 'class_weight': 'balanced'}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(**params))
    ])
    base_pipe.set_output(transform="pandas")

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- Random Forest Classification Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"F1 Score (Train): {f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"F1 Score (Test): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    print("-" * 35)
    return best_model

#---Function:xgboost_classification---
def xgboost_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, **model_params):
    """
    XGBoost Classifier with Gradient Boosting optimization.
    """
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    counts = y_train.value_counts()
    ratio = counts.iloc[0] / counts.iloc[1] if len(counts) == 2 else 1

    params = {'random_state': 42, 'eval_metric': 'logloss', 'scale_pos_weight': ratio}
    params.update(model_params)

    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBClassifier(**params))
    ])
    base_pipe.set_output(transform="pandas")

    if for_stacking: return base_pipe

    param_grid = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    print(f"--- XGBoost Classification Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"F1 Score (Train): {f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"F1 Score (Test): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    print("-" * 35)
    return best_model
