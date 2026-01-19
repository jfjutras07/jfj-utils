import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, classification_report

#---Function:stacking_classification_ensemble---
def stacking_classification_ensemble(train_df, test_df, outcome, predictors, base_estimators, final_estimator=None, cv=5, **meta_params):
    """
    Orchestrates a Stacking Ensemble for Classification.
    Combines multiple classifiers using a meta-model (default: Logistic Regression).
    """
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    # Initialize default meta-model if none provided, applying meta_params
    if final_estimator is None:
        final_estimator = LogisticRegression(**meta_params)

    # Initialize Stacking
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=-1,
        passthrough=False
    )

    print(f"Starting Classification Stacking fit with {len(base_estimators)} base models...")
    stacking_model.fit(X_train, y_train)

    # Evaluation
    y_pred = stacking_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n--- Stacking Classification Final Summary ---")
    print(f"Accuracy Score: {acc:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 35)

    return stacking_model

#---Function:stacking_regression_ensemble---
def stacking_regression_ensemble(train_df, test_df, outcome, predictors, base_estimators, final_estimator=None, cv=5, **meta_params):
    """
    Orchestrates a Stacking Ensemble for Regression.
    Combines multiple regressors using a meta-model (default: RidgeCV).
    """
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    # Initialize default meta-model (RidgeCV) with meta_params if applicable
    if final_estimator is None:
        # Note: RidgeCV specific params can be passed via meta_params
        final_estimator = RidgeCV(**meta_params)

    # Initialize Stacking
    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=-1,
        passthrough=False
    )

    print(f"Starting Regression Stacking fit with {len(base_estimators)} base models...")
    stacking_model.fit(X_train, y_train)

    # Evaluation
    y_pred = stacking_model.predict(X_test)
    
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    print(f"\n--- Stacking Regression Final Summary ---")
    print(f"R2 Score: {metrics['R2']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print("-" * 35)

    return stacking_model
