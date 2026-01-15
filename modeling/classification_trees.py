import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from IPython.display import display

#---Function:catboost_classification---
def catboost_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    CatBoost Classifier with GridSearchCV.
    English comment: CatBoost handles categorical features efficiently using symmetric trees.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', CatBoostClassifier(random_state=42, verbose=0))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {
        'model__iterations': [100, 200],
        'model__depth': [4, 6],
        'model__learning_rate': [0.05, 0.1]
    }
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"--- CatBoost Optimized (F1: {grid_search.best_score_:.4f}) ---")
    return grid_search.best_estimator_

#---Function:compare_classification_tree_models---
def compare_classification_tree_models(train_df, test_df, outcome, predictors, cv=5):
    """
    Executes and compares tree-based models, sorted by alphabetical order of execution.
    English comment: Final comparison based on weighted F1-score to handle potential class imbalance.
    """
    print(f"Starting Tree Models Comparison | Predictors: {len(predictors)}")
    print("-" * 45)

    # Dictionary to store results
    results = {}
    
    # Executing in alphabetical order
    print("\n[1/5] Running CatBoost...")
    results['CatBoost'] = catboost_classification(train_df, test_df, outcome, predictors, cv)
    
    print("\n[2/5] Running Decision Tree...")
    results['DecisionTree'] = decision_tree_classification(train_df, test_df, outcome, predictors, cv)
    
    print("\n[3/5] Running LightGBM...")
    results['LightGBM'] = lightgbm_classification(train_df, test_df, outcome, predictors, cv)
    
    print("\n[4/5] Running Random Forest...")
    results['RandomForest'] = random_forest_classification(train_df, test_df, outcome, predictors, cv)
    
    print("\n[5/5] Running XGBoost...")
    results['XGBoost'] = xgboost_classification(train_df, test_df, outcome, predictors, cv)

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
    
    # English comment: Extracting feature importance from the winning tree model
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
    English comment: Simple partitioning model, prone to overfitting if depth is not tuned.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', DecisionTreeClassifier(random_state=42))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__max_depth': [None, 5, 10, 20], 'model__min_samples_leaf': [1, 5, 10]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"--- Decision Tree Optimized (F1: {grid_search.best_score_:.4f}) ---")
    return grid_search.best_estimator_

#---Function:lightgbm_classification---
def lightgbm_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    LightGBM Classifier with leaf-wise growth optimization.
    English comment: Efficient gradient boosting framework that uses tree-based learning algorithms.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', LGBMClassifier(random_state=42, verbosity=-1))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__n_estimators': [100, 300], 'model__learning_rate': [0.01, 0.1], 'model__num_leaves': [31, 50]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"--- LightGBM Optimized (F1: {grid_search.best_score_:.4f}) ---")
    return grid_search.best_estimator_

#---Function:random_forest_classification---
def random_forest_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Random Forest Classifier using Bagging technique.
    English comment: Ensemble of decision trees to reduce variance and improve robustness.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(random_state=42))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10, 20]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"--- Random Forest Optimized (F1: {grid_search.best_score_:.4f}) ---")
    return grid_search.best_estimator_

#---Function:xgboost_classification---
def xgboost_classification(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    XGBoost Classifier with Gradient Boosting optimization.
    English comment: Advanced implementation of gradient boosting with built-in regularization.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    param_grid = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1], 'model__max_depth': [3, 6]}
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"--- XGBoost Optimized (F1: {grid_search.best_score_:.4f}) ---")
    return grid_search.best_estimator_
