import numpy as np
import pandas as pd
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

#---Function:catboost_classification---
def catboost_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    CatBoost Classifier with GridSearchCV.
    Handles categorical features efficiently using symmetric trees and gradient boosting.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', CatBoostClassifier(random_state=42, verbose=0))
    ])
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

#---Function:compare_classification_tree_models---
def compare_classification_tree_models(train_df, test_df, outcome, predictors, cv=5):
    """
    Executes and compares tree-based models, sorted by alphabetical order.
    Final comparison based on weighted F1-score to evaluate performance.
    """
    print(f"Starting Tree Models Comparison | Predictors: {len(predictors)}")
    print("-" * 45)

    results = {
        'CatBoost': catboost_classification(train_df, test_df, outcome, predictors, cv),
        'DecisionTree': decision_tree_classification(train_df, test_df, outcome, predictors, cv),
        'LightGBM': lightgbm_classification(train_df, test_df, outcome, predictors, cv),
        'RandomForest': random_forest_classification(train_df, test_df, outcome, predictors, cv),
        'XGBoost': xgboost_classification(train_df, test_df, outcome, predictors, cv)
    }

    X_test, y_test = test_df[predictors], test_df[outcome]
    perf_metrics = []

    for name, model in results.items():
        y_pred = model.predict(X_test)
        perf_metrics.append({
            "Model": name,
            "F1_Weighted": f1_score(y_test, y_pred, average='weighted'),
            "Accuracy": accuracy_score(y_test, y_pred)
        })

    comparison_df = pd.DataFrame(perf_metrics).set_index("Model").sort_values(by="F1_Weighted", ascending=False)
    
    print("\n--- Final Tree Comparison (Sorted by F1) ---")
    print(comparison_df.to_string())
    
    winner_name = comparison_df.index[0]
    winner_model = results[winner_name]
    
    print(f"\nCHAMPION: {winner_name}")
    
    actual_model = winner_model.named_steps['model']
    feat_imp = pd.DataFrame({
        'Feature': predictors,
        'Importance': actual_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)
    
    print("\nTop 10 Feature Importances:")
    display(feat_imp)
    return winner_model

#---Function:decision_tree_classification---
def decision_tree_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Decision Tree Classifier with depth optimization.
    Simple partitioning model used as a baseline for more complex tree ensembles.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', DecisionTreeClassifier(random_state=42))
    ])
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
def lightgbm_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    LightGBM Classifier with leaf-wise growth optimization.
    High-performance gradient boosting framework designed for speed and efficiency.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', LGBMClassifier(random_state=42, verbosity=-1))
    ])
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
def random_forest_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Random Forest Classifier using Bagging technique.
    Ensemble of decision trees that reduces variance through bootstrap aggregating.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(random_state=42))
    ])
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
def xgboost_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    XGBoost Classifier with Gradient Boosting optimization.
    Advanced implementation of gradient boosting with built-in regularization to prevent overfitting.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

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
