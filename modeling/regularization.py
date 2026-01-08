from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from IPython.display import display
import numpy as np
import pandas as pd

#--- Function : compare_regularized_models ---
def compare_regularized_models(train_df, test_df, outcome, predictors, cv=5):
    """
    Executes and compares Lasso, Ridge, and ElasticNet regressions.
    Displays logs, summary table, and non-zero coefficients for the champion.
    """
    import pandas as pd
    from IPython.display import display

    print("Starting Regularized Models Comparison...")
    print(f"Predictors: {len(predictors)} | CV Folds: {cv}")
    print("-" * 35)

    #Execute individual regressions
    lasso_res = lasso_regression(train_df, test_df, outcome, predictors, cv=cv)
    ridge_res = ridge_regression(train_df, test_df, outcome, predictors, cv=cv)
    enet_res = elasticnet_regression(train_df, test_df, outcome, predictors, cv=cv)

    #Compile metrics for ranking
    comparison_data = {
        "Lasso": lasso_res["metrics"],
        "Ridge": ridge_res["metrics"],
        "ElasticNet": enet_res["metrics"]
    }
    comparison_df = pd.DataFrame(comparison_data).T.sort_values(by="R2", ascending=False)

    #Print Final Comparison Table
    print("\n--- Final Comparison Summary ---")
    print(comparison_df[["R2", "MAE", "RMSE"]].to_string())
    print("-" * 35)

    #Extract and sort non-zero coefficients for the champion
    winner_name = comparison_df.index[0]
    all_results = {"lasso": lasso_res, "ridge": ridge_res, "elasticnet": enet_res}
    winner_data = all_results[winner_name.lower()]
    
    coeffs = winner_data['coefficients'].copy()
    #Filter non-zero and sort by absolute impact
    active_coeffs = coeffs[coeffs['Coefficient'] != 0].copy()
    active_coeffs['Abs_Coefficient'] = active_coeffs['Coefficient'].abs()
    #Limit to Top 5
    active_coeffs = active_coeffs.sort_values(by='Abs_Coefficient', ascending=False).head(5).drop(columns=['Abs_Coefficient'])
    
    print(f"\nModel Champion: {winner_name}")
    print(f"Top 5 Coefficients for {winner_name} (Sorted by impact):")
    display(active_coeffs)
    
#--- Function : elasticnet_regression ---
def elasticnet_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    Perform ElasticNet regression with automated Cross-Validation.

    When to use:
    - When there are multiple correlated predictors (grouping effect).
    - When you want a compromise between Lasso (feature selection) and Ridge (stability).
    - Ideal for complex datasets where the exact nature of regularization is unknown.

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
        Dictionary containing the fitted model, best alpha/l1_ratio, metrics, and coefficients.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    #Data extraction
    X_train = train_df[predictors]
    y_train = train_df[outcome]
    X_test = test_df[predictors]
    y_test = test_df[outcome]

    #Model fitting with built-in Cross-Validation
    #l1_ratio: 1 is Lasso, 0.5 is equal mix, near 0 is Ridge.
    #ElasticNetCV tests a grid of alphas and l1_ratios to find the best balance.
    model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=cv, random_state=42).fit(X_train, y_train)

    #Predictions and Evaluation
    y_pred = model.predict(X_test)
    
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Best_Alpha": model.alpha_,
        "Best_L1_Ratio": model.l1_ratio_
    }

    #Coefficients summary
    coef_df = pd.DataFrame({
        'Feature': predictors,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)

    #Filtering features kept by the model (non-zero)
    active_features = coef_df[coef_df['Coefficient'] != 0]

    #Print summary
    print(f"--- ElasticNet Regression Summary ---")
    print(f"Best Alpha: {metrics['Best_Alpha']:.5f}")
    print(f"Best L1 Ratio: {metrics['Best_L1_Ratio']:.2f}")
    print(f"R2 Score (Test): {metrics['R2']:.4f}")
    print(f"MAE (Test): {metrics['MAE']:.4f}")
    print(f"Features Retained: {len(active_features)} / {len(predictors)}")
    print("-" * 35)

    return {
        "model": model,
        "metrics": metrics,
        "coefficients": coef_df,
        "active_features": active_features
    }

#--- Function : lasso_regression ---
def lasso_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    Perform Lasso regression with automated Cross-Validation for alpha selection.

    When to use:
    - High-dimensional datasets where feature selection is required.
    - To prevent overfitting by shrinking non-significant coefficients to zero.
    - When interpretability and model parsimony are priorities.

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
        Dictionary containing the fitted model, best alpha, metrics, and coefficient importance.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    #Data extraction
    X_train = train_df[predictors]
    y_train = train_df[outcome]
    X_test = test_df[predictors]
    y_test = test_df[outcome]

    #Model fitting with built-in Cross-Validation
    #LassoCV explores a grid of alphas and selects the one minimizing the MSE
    model = LassoCV(cv=cv, random_state=42).fit(X_train, y_train)

    #Predictions and Evaluation
    y_pred = model.predict(X_test)
    
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Best_Alpha": model.alpha_
    }

    #Coefficients summary
    coef_df = pd.DataFrame({
        'Feature': predictors,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)

    #Filtering features kept by the model (non-zero)
    active_features = coef_df[coef_df['Coefficient'] != 0]

    #Print summary
    print(f"--- Lasso Regression Summary ---")
    print(f"Best Alpha (Optimal Penalty): {metrics['Best_Alpha']:.5f}")
    print(f"R2 Score (Test): {metrics['R2']:.4f}")
    print(f"MAE (Test): {metrics['MAE']:.4f}")
    print(f"Features Retained: {len(active_features)} / {len(predictors)}")
    print("-" * 35)

    return {
        "model": model,
        "metrics": metrics,
        "coefficients": coef_df,
        "active_features": active_features
    }

#--- Function : ridge_regression ---
def ridge_regression(train_df, test_df, outcome, predictors, cv=5):
    """
    Perform Ridge regression with automated Cross-Validation for alpha selection.

    When to use:
    - Predictors are highly correlated (multicollinearity).
    - To prevent overfitting while keeping all variables in the model.
    - When you believe most features contribute at least slightly to the outcome.

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
        Dictionary containing the fitted model, best alpha, metrics, and coefficients.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    #Data extraction
    X_train = train_df[predictors]
    y_train = train_df[outcome]
    X_test = test_df[predictors]
    y_test = test_df[outcome]

    #Model fitting with built-in Cross-Validation
    #We provide a range of alphas to explore via RidgeCV
    alphas = np.logspace(-3, 3, 100)
    model = RidgeCV(alphas=alphas, cv=cv).fit(X_train, y_train)

    #Predictions and Evaluation
    y_pred = model.predict(X_test)
    
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Best_Alpha": model.alpha_
    }

    #Coefficients summary
    coef_df = pd.DataFrame({
        'Feature': predictors,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)

    #Print summary
    print(f"--- Ridge Regression Summary ---")
    print(f"Best Alpha (Optimal Penalty): {metrics['Best_Alpha']:.5f}")
    print(f"R2 Score (Test): {metrics['R2']:.4f}")
    print(f"MAE (Test): {metrics['MAE']:.4f}")
    print("-" * 35)

    return {
        "model": model,
        "metrics": metrics,
        "coefficients": coef_df
    }
