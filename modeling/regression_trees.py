from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

#--- Function : random_forest_regression ---
def random_forest_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    Perform Random Forest regression with GridSearch for hyperparameter tuning.

    When to use:
    - To capture non-linear relationships and complex feature interactions.
    - When the dataset contains a mix of categorical and numerical features.
    - To provide a robust baseline that handles outliers and noise effectively.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset containing outcome and predictors.
    test_df : pd.DataFrame
        Testing dataset for final performance evaluation.
    outcome : str
        Dependent variable (target).
    predictors : list of str
        List of predictor variables.
    cv : int, default 5
        Number of folds for Cross-Validation.

    Returns:
    --------
    results : dict
        Dictionary containing the best model, params, metrics, and feature importance.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    #Data extraction
    X_train = train_df[predictors]
    y_train = train_df[outcome]
    X_test = test_df[predictors]
    y_test = test_df[outcome]

    #Define the parameter grid to tune
    #n_estimators: number of trees
    #max_depth: complexity of the trees
    #min_samples_split: minimum samples required to split a node
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }

    #Initialize and fit GridSearchCV
    #n_jobs=-1 uses all available processors for speed
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    #Best model and predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Best_Params": grid_search.best_params_
    }

    #Feature importance summary (MDI - Mean Decrease in Impurity)
    importance_df = pd.DataFrame({
        'Feature': predictors,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    #Print summary
    print(f"--- Random Forest Summary ---")
    print(f"Best Params: {metrics['Best_Params']}")
    print(f"R2 Score (Test): {metrics['R2']:.4f}")
    print(f"MAE (Test): {metrics['MAE']:.4f}")
    print(f"Top 3 Features: {importance_df['Feature'].iloc[:3].tolist()}")
    print("-" * 35)

    return {
        "model": best_model,
        "metrics": metrics,
        "importance": importance_df
    }
