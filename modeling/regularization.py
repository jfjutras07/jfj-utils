import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from IPython.display import display

#--- Function : compare_regularized_models ---
def compare_regularized_models(train_df, test_df, outcome, predictors, model_type='classification', cv=5):
    """
    Executes and compares Lasso, Ridge, and ElasticNet.
    Adapts metrics and models based on model_type ('classification' or 'regression').
    """
    print(f"Starting Regularized Models Comparison | Mode: {model_type.upper()} | Predictors: {len(predictors)}")
    print("-" * 45)

    # Dictionary to store results
    results = {
        "ElasticNet": elasticnet_model(train_df, test_df, outcome, predictors, model_type, cv=cv, for_compare=True),
        "Lasso": lasso_model(train_df, test_df, outcome, predictors, model_type, cv=cv, for_compare=True),
        "Ridge": ridge_model(train_df, test_df, outcome, predictors, model_type, cv=cv, for_compare=True)
    }

    comparison_list = []
    X_test, y_test = test_df[predictors], test_df[outcome]
    
    # Define primary metric for sorting
    main_metric = "ROC_AUC" if model_type == 'classification' else "R2"

    for name, data in results.items():
        model = data['model']
        y_pred = model.predict(X_test)
        
        res = {"Model": name, "Alpha/C": round(data['alpha'], 5)}
        
        if model_type == 'classification':
            y_proba = model.predict_proba(X_test)[:, 1]
            res.update({
                "ROC_AUC": roc_auc_score(y_test, y_proba),
                "Recall": recall_score(y_test, y_pred),
                "Accuracy": accuracy_score(y_test, y_pred)
            })
        else:
            res.update({
                "R2": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
            })
        comparison_list.append(res)

    comparison_df = pd.DataFrame(comparison_list).set_index("Model").sort_values(by=main_metric, ascending=False)

    print(f"\n--- Final Regularized Comparison (Sorted by {main_metric}) ---")
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

#--- Function : elasticnet_model ---
def elasticnet_model(train_df, test_df, outcome, predictors, model_type='classification', cv=5, for_stacking=False, for_compare=False, **model_params):
    """
    Perform ElasticNet (L1 + L2).
    """
    if model_type == 'classification':
        params = {
            'penalty': 'elasticnet', 
            'l1_ratios': [.1, .5, .9], 
            'solver': 'saga', 
            'cv': cv, 
            'class_weight': 'balanced', 
            'random_state': 42, 
            'max_iter': 2000
        }
        params.update(model_params)
        model_obj = LogisticRegressionCV(**params)
    else:
        params = {'l1_ratio': [.1, .5, .7, .9, 1], 'cv': cv, 'random_state': 42}
        params.update(model_params)
        model_obj = ElasticNetCV(**params)

    return _fit_and_report(model_obj, train_df, test_df, outcome, predictors, "ElasticNet", model_type, for_stacking, for_compare)

#--- Function : lasso_model ---
def lasso_model(train_df, test_df, outcome, predictors, model_type='classification', cv=5, for_stacking=False, for_compare=False, **model_params):
    """
    Perform Lasso (L1).
    """
    if model_type == 'classification':
        params = {'penalty': 'l1', 'solver': 'liblinear', 'cv': cv, 'class_weight': 'balanced', 'random_state': 42}
        params.update(model_params)
        model_obj = LogisticRegressionCV(**params)
    else:
        params = {'cv': cv, 'random_state': 42}
        params.update(model_params)
        model_obj = LassoCV(**params)

    return _fit_and_report(model_obj, train_df, test_df, outcome, predictors, "Lasso", model_type, for_stacking, for_compare)

#--- Function : ridge_model ---
def ridge_model(train_df, test_df, outcome, predictors, model_type='classification', cv=5, for_stacking=False, for_compare=False, **model_params):
    """
    Perform Ridge (L2).
    """
    if model_type == 'classification':
        params = {'penalty': 'l2', 'cv': cv, 'class_weight': 'balanced', 'random_state': 42}
        params.update(model_params)
        model_obj = LogisticRegressionCV(**params)
    else:
        params = {'alphas': np.logspace(-3, 3, 10), 'cv': cv}
        params.update(model_params)
        model_obj = RidgeCV(**params)

    return _fit_and_report(model_obj, train_df, test_df, outcome, predictors, "Ridge", model_type, for_stacking, for_compare)

#--- Internal Helper : _fit_and_report ---
def _fit_and_report(model_obj, train_df, test_df, outcome, predictors, name, model_type, for_stacking, for_compare):
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model_obj)
    ])

    if for_stacking: return base_pipe

    X_train, y_train = train_df[predictors], train_df[outcome]
    X_test, y_test = test_df[predictors], test_df[outcome]

    base_pipe.fit(X_train, y_train)
    fitted_model = base_pipe.named_steps['model']
    
    # Get alpha/C and coefficients
    alpha = getattr(fitted_model, 'alpha_', getattr(fitted_model, 'C_', [0])[0])
    if isinstance(alpha, np.ndarray): alpha = alpha[0]
    
    raw_coefs = fitted_model.coef_
    coefs = raw_coefs[0] if raw_coefs.ndim > 1 else raw_coefs
    coef_df = pd.DataFrame({'Feature': predictors, 'Coefficient': coefs})

    if not for_compare:
        print(f"--- {name} {model_type.capitalize()} Summary ---")
        y_pred = base_pipe.predict(X_test)
        if model_type == 'classification':
            y_proba = base_pipe.predict_proba(X_test)[:, 1]
            print(f"ROC_AUC (Test): {roc_auc_score(y_test, y_proba):.4f}")
            print(f"Recall (Test): {recall_score(y_test, y_pred):.4f}")
        else:
            print(f"R2 Score (Test): {r2_score(y_test, y_pred):.4f}")
        print("-" * 35)

    return {"model": base_pipe, "alpha": alpha, "coefficients": coef_df}
