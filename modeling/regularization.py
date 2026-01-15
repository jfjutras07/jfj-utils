import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from IPython.display import display

#--- Function : compare_regularized_models ---
def compare_regularized_models(train_df, test_df, outcome, predictors, cv=5):
    """
    Executes and compares Lasso, Ridge, and ElasticNet regressions.
    Displays logs, summary table, and non-zero coefficients for the champion.
    """
    print(f"Starting Regularized Models Comparison | Predictors: {len(predictors)}")
    print("-" * 45)

    # Dictionary to store results for ranking
    results = {
        "ElasticNet": elasticnet_regression(train_df, test_df, outcome, predictors, cv=cv, for_compare=True),
        "Lasso": lasso_regression(train_df, test_df, outcome, predictors, cv=cv, for_compare=True),
        "Ridge": ridge_regression(train_df, test_df, outcome, predictors, cv=cv, for_compare=True)
    }

    comparison_list = []
    X_test, y_test = test_df[predictors], test_df[outcome]

    for name, data in results.items():
        y_pred = data['model'].predict(X_test)
        comparison_list.append({
            "Model": name,
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "Alpha": round(data['alpha'], 5)
        })

    comparison_df = pd.DataFrame(comparison_list).set_index("Model").sort_values(by="R2", ascending=False)

    print("\n--- Final Regularized Comparison (Sorted by R2) ---")
    print(comparison_df.to_string())
    print("-" * 45)

    # Extract champion data
    winner_name = comparison_df.index[0]
    winner_data = results[winner_name]
    
    print(f"\nCHAMPION: {winner_name}")
    coeffs = winner_data['coefficients']
    active_coeffs = coeffs[coeffs['Coefficient'] != 0].copy()
    active_coeffs['Abs_Impact'] = active_coeffs['Coefficient'].abs()
    
    print(f"Top 5 Coefficients for {winner_name}:")
    display(active_coeffs.sort_values(by='Abs_Impact', ascending=False).head(5).drop(columns=['Abs_Impact']))
    
    return winner_data["model"]

#--- Function : elasticnet_regression ---
def elasticnet_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, for_compare=False):
    """
    Perform ElasticNet regression with automated L1/L2 ratio tuning.
    Combination of Lasso and Ridge penalties.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=cv, random_state=42))
    ])

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    base_pipe.fit(X_train, y_train)
    fitted_model = base_pipe.named_steps['model']
    
    y_pred_train = base_pipe.predict(X_train)
    y_pred_test = base_pipe.predict(X_test)
    
    coef_df = pd.DataFrame({'Feature': predictors, 'Coefficient': fitted_model.coef_})
    active_count = len(coef_df[coef_df['Coefficient'] != 0])

    if not for_compare:
        print(f"--- ElasticNet Regression Summary ---")
        print(f"Best Alpha: {fitted_model.alpha_:.5f} | L1 Ratio: {fitted_model.l1_ratio_:.2f}")
        print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
        print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f} | Active Features: {active_count}/{len(predictors)}")
        print("-" * 35)

    return {"model": base_pipe, "alpha": fitted_model.alpha_, "coefficients": coef_df}

#--- Function : lasso_regression ---
def lasso_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, for_compare=False):
    """
    Perform Lasso regression (L1).
    Excellent for automated feature selection by shrinking coefficients to zero.
    """
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LassoCV(cv=cv, random_state=42))
    ])

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    base_pipe.fit(X_train, y_train)
    fitted_model = base_pipe.named_steps['model']
    
    y_pred_train = base_pipe.predict(X_train)
    y_pred_test = base_pipe.predict(X_test)
    
    coef_df = pd.DataFrame({'Feature': predictors, 'Coefficient': fitted_model.coef_})
    active_count = len(coef_df[coef_df['Coefficient'] != 0])

    if not for_compare:
        print(f"--- Lasso Regression Summary ---")
        print(f"Best Alpha: {fitted_model.alpha_:.5f}")
        print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
        print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f} | Active Features: {active_count}/{len(predictors)}")
        print("-" * 35)

    return {"model": base_pipe, "alpha": fitted_model.alpha_, "coefficients": coef_df}

#--- Function : ridge_regression ---
def ridge_regression(train_df, test_df, outcome, predictors, cv=5, for_stacking=False, for_compare=False):
    """
    Perform Ridge regression (L2).
    Best for handling multicollinearity by shrinking coefficients proportionally.
    """
    alphas = np.logspace(-3, 3, 100)
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RidgeCV(alphas=alphas, cv=cv))
    ])

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    base_pipe.fit(X_train, y_train)
    fitted_model = base_pipe.named_steps['model']
    
    y_pred_train = base_pipe.predict(X_train)
    y_pred_test = base_pipe.predict(X_test)
    
    coef_df = pd.DataFrame({'Feature': predictors, 'Coefficient': fitted_model.coef_})

    if not for_compare:
        print(f"--- Ridge Regression Summary ---")
        print(f"Best Alpha: {fitted_model.alpha_:.5f}")
        print(f"R2 Score (Train): {r2_score(y_train, y_pred_train):.4f}")
        print(f"R2 Score (Test): {r2_score(y_test, y_pred_test):.4f}")
        print("-" * 35)

    return {"model": base_pipe, "alpha": fitted_model.alpha_, "coefficients": coef_df}
