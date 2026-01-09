import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#---Function:stacking_ensemble---
def stacking_ensemble(train_df, test_df, outcome, predictors, base_estimators, final_estimator=None, cv=5):
    """
    Orchestrates a Stacking Ensemble.
    """
    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    #If no meta-model is provided, RidgeCV is the gold standard for stacking
    if final_estimator is None:
        final_estimator = RidgeCV()

    #Initialize Stacking
    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=-1,
        passthrough=False
    )

    print(f"Starting Stacking fit with {len(base_estimators)} base models...")
    stacking_model.fit(X_train, y_train)

    #Evaluation
    y_pred = stacking_model.predict(X_test)
    
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    print(f"\n--- Stacking Ensemble Final Summary ---")
    print(f"R2 Score: {metrics['R2']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print("-" * 35)

    return stacking_model
