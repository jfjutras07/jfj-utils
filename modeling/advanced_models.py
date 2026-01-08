from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from IPython.display import display

def svm_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    Support Vector Regression (SVR) with GridSearchCV and Permutation Importance.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    #Hyperparameter tuning
    param_grid = {
        'kernel': ['rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2]
    }

    svr = SVR()
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    #Metrics
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    #Permutation Importance (Standard way to get importance for SVR)
    perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
    
    importance_df = pd.DataFrame({
        'Feature': predictors,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False).head(5)

    print(f"--- SVM (SVR) Summary ---")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"R2 Score (Test): {metrics['R2']:.4f}")
    print(f"MAE (Test): {metrics['MAE']:.4f}")
    print(f"Top 3 Features: {importance_df['Feature'].iloc[:3].tolist()}")
    print("-" * 35)

    print("\nPermutation Importance (Sorted by impact):")
    display(importance_df)
