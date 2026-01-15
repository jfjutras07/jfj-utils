import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

#---Function:logistic_regression---
def logistic_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Logistic Regression for binary or multiclass classification.
    Optimizes regularization strength (C) and penalty type using GridSearchCV.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced'))
    ])
    
    if for_stacking: return base_pipe
        
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]
    
    param_grid = {
        'model__C': [0.1, 1.0, 10.0],
        'model__penalty': ['l1', 'l2']
    }
    
    grid_search = GridSearchCV(base_pipe, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    print(f"--- Logistic Regression Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"F1 Score (Train): {f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"F1 Score (Test): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    
    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, y_pred_test))
    print("-" * 35)
    
    return best_model
