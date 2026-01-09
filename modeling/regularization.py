from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from IPython.display import display
import numpy as np
import pandas as pd

#--- Function : compare_regularized_models ---
def compare_regularized_models(train_df, test_df, outcome, predictors, cv=5):
    """
    Executes and compares Lasso, Ridge, and ElasticNet regressions.
    Displays logs, summary table, and non-zero coefficients for the champion.
    """
    print("Starting Regularized Models Comparison...")
    print(f"Predictors: {len(predictors)} | CV Folds: {cv}")
    print("-" * 35)

    # Execute individual regressions (Solo mode)
    lasso_res = lasso_regression(train_df, test_df, outcome, predictors, cv=cv)
    ridge_res = ridge_regression(train_df, test_df, outcome, predictors, cv=cv)
    enet_res = elasticnet_regression(train_df, test_df, outcome, predictors, cv=cv)

    # Compile metrics for ranking
    comparison_data = {
        "Lasso": lasso_res["metrics"],
        "Ridge": ridge_res["metrics"],
        "ElasticNet": enet_res["metrics"]
    }
    comparison_df = pd.DataFrame(comparison_data).T.sort_values(by="R2", ascending=False)

    # Print Final Comparison Table
    print("\n--- Final Comparison Summary ---")
    print(comparison_df[["R2", "MAE", "RMSE"]].to_string())
    print("-" * 35)

    # Extract champion data
    winner_name = comparison_df.index[0]
    all_results = {"lasso": lasso_res, "ridge": ridge_res, "elasticnet": enet_res}
    winner_data = all_results[winner_name.lower()]
    
    coeffs = winner_data['coefficients'].copy()
    active_coeffs = coeffs[coeffs['Coefficient'] != 0].copy()
    active_coeffs['Abs_Coefficient'] = active_coeffs['Coefficient'].abs()
    active_coeffs = active_coeffs.sort_values(by='Abs_Coefficient', ascending=False).head(5).drop(columns=['Abs_Coefficient'])
    
    print(f"\nModel Champion: {winner_name}")
    print(f"Top 5 Coefficients for {winner_name} (Sorted by impact):")
    display(active_coeffs)
    
    return winner_data["model"]

#--- Function : elasticnet_regression ---
def elasticnet_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Perform ElasticNet regression with automated Cross-Validation.
    """
    model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=cv, random_state=42)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    if for_stacking:
        return pipeline

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    fitted_model = pipeline.named_steps['model']
    
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Best_Alpha": fitted_model.alpha_,
        "Best_L1_Ratio": fitted_model.l1_ratio_
    }

    coef_df = pd.DataFrame({'Feature': predictors, 'Coefficient': fitted_model.coef_}).sort_values(by='Coefficient', ascending=False)
    active_features = coef_df[coef_df['Coefficient'] != 0]

    print(f"--- ElasticNet Regression Summary ---")
    print(f"Best Alpha: {metrics['Best_Alpha']:.5f} | Best L1: {metrics['Best_L1_Ratio']:.2f}")
    print(f"R2 Score (Test): {metrics['R2']:.4f} | Features: {len(active_features)}/{len(predictors)}")
    print("-" * 35)

    return {"model": pipeline, "metrics": metrics, "coefficients": coef_df, "active_features": active_features}

#--- Function : lasso_regression ---
def lasso_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Perform Lasso regression with automated Cross-Validation for feature selection.
    """
    model = LassoCV(cv=cv, random_state=42)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    if for_stacking:
        return pipeline

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    fitted_model = pipeline.named_steps['model']
    
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Best_Alpha": fitted_model.alpha_
    }

    coef_df = pd.DataFrame({'Feature': predictors, 'Coefficient': fitted_model.coef_}).sort_values(by='Coefficient', ascending=False)
    active_features = coef_df[coef_df['Coefficient'] != 0]

    print(f"--- Lasso Regression Summary ---")
    print(f"Best Alpha: {metrics['Best_Alpha']:.5f}")
    print(f"R2 Score (Test): {metrics['R2']:.4f} | Features: {len(active_features)}/{len(predictors)}")
    print("-" * 35)

    return {"model": pipeline, "metrics": metrics, "coefficients": coef_df, "active_features": active_features}

#--- Function : ridge_regression ---
def ridge_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False):
    """
    Perform Ridge regression with automated Cross-Validation for multicollinearity.
    """
    alphas = np.logspace(-3, 3, 100)
    model = RidgeCV(alphas=alphas, cv=cv)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    if for_stacking:
        return pipeline

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    fitted_model = pipeline.named_steps['model']
    
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Best_Alpha": fitted_model.alpha_
    }

    coef_df = pd.DataFrame({'Feature': predictors, 'Coefficient': fitted_model.coef_}).sort_values(by='Coefficient', ascending=False)

    print(f"--- Ridge Regression Summary ---")
    print(f"Best Alpha: {metrics['Best_Alpha']:.5f}")
    print(f"R2 Score (Test): {metrics['R2']:.4f}")
    print("-" * 35)

    return {"model": pipeline, "metrics": metrics, "coefficients": coef_df}
