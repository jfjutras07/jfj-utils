import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

#--- Function : fit_regularized_models ---
def fit_regularized_models(
    X_train,
    y_train,
    X_test,
    y_test,
    n_splits=5,
    random_state=42,
):
    """
    Fit and compare Ridge, Lasso, and Elastic Net regression models with hyperparameter tuning.
    This function automatically handles multicollinearity and multi-output targets.

    Parameters
    ----------
    X_train : array-like or DataFrame
        Training features.
    y_train : array-like or DataFrame
        Training target(s).
    X_test : array-like or DataFrame
        Test features.
    y_test : array-like or DataFrame
        Test target(s).
    n_splits : int, optional
        Number of folds for KFold cross-validation.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        {
            "models": { "ridge": ..., "lasso": ..., "elasticnet": ... },
            "scores": DataFrame summary of performance,
            "best_model_name": str,
            "best_model": fitted model instance
        }
    """

    #Custom RMSE scorer for GridSearchCV
    def rmse_scorer(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    rmse_sklearn = make_scorer(rmse_scorer, greater_is_better=False)

    #Cross-validation setup
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    #Define models and grids
    models = {
        "ridge": {
            "estimator": Ridge(),
            "param_grid": {"alpha": [0.01, 0.1, 1.0, 10, 50, 100]},
        },
        "lasso": {
            "estimator": Lasso(max_iter=5000),
            "param_grid": {"alpha": [0.001, 0.01, 0.1, 1.0, 10]},
        },
        "elasticnet": {
            "estimator": ElasticNet(max_iter=5000),
            "param_grid": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10],
                "l1_ratio": [0.1, 0.5, 0.7, 0.9],
            },
        },
    }

    fitted_models = {}
    results = []

    #Loop through each regularized model
    for name, cfg in models.items():
        grid = GridSearchCV(
            cfg["estimator"],
            cfg["param_grid"],
            scoring=rmse_sklearn,
            cv=cv,
            n_jobs=-1,
        )

        #MultiOutputRegressor wrapper if y_train has multiple columns
        model = MultiOutputRegressor(grid)
        model.fit(X_train, y_train)

        #Predictions
        y_pred = model.predict(X_test)

        #Store results
        best_params = model.estimators_[0].best_params_
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        fitted_models[name] = model

        results.append(
            {
                "model": name,
                "best_params": best_params,
                "rmse": rmse,
                "r2": r2,
            }
        )

    #Convert score summary to DataFrame
    scores_df = pd.DataFrame(results)

    #Determine best model by RMSE (lower is better)
    best_model_name = scores_df.sort_values("rmse").iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    return {
        "models": fitted_models,
        "scores": scores_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
    }
