import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
from sklearn.inspection import permutation_importance, PartialDependenceDisplay, partial_dependence

#---Function:interaction_effects---
def interaction_effects(model, test_df, predictors, top_n=5):
    """
    Identify interactions by comparing 2D Partial Dependence vs 1D.
    Higher values indicate that features impact the outcome more when combined.
    """
    # Select features
    X_test = test_df[predictors]
    
    # Target: First column logic
    y_test = test_df.iloc[:, 0]
    
    # Run permutation importance 
    # Removed random_state to ensure compatibility with older sklearn versions
    perm_res = permutation_importance(
        model, 
        X_test, 
        y_test, 
        n_repeats=5
    )
    
    # Get indices of top importance mean
    top_indices = perm_res.importances_mean.argsort()[-top_n:]
    top_feats = [predictors[i] for i in top_indices]
    
    print(f"--- Top Interaction Analysis for: {top_feats} ---")
    
    if len(top_feats) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        # Display interaction between the top two features
        PartialDependenceDisplay.from_estimator(
            model, 
            X_test, 
            [(top_feats[-1], top_feats[-2])], 
            ax=ax
        )
        plt.show()
        
    return top_feats

#---Function:lime_analysis---
def lime_analysis(model, train_df, test_df, predictors, row_index=0):
    """
    Local Interpretable Model-agnostic Explanations (LIME).
    Explains a specific individual prediction by fitting a local linear model.
    """
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=train_df[predictors].values,
        feature_names=predictors,
        mode='regression'
    )
    
    exp = explainer.explain_instance(
        data_row=test_df[predictors].iloc[row_index].values,
        predict_fn=model.predict
    )
    
    print(f"--- LIME Explanation for Row {row_index} ---")
    return exp

#---Function:pdp_plots---
def pdp_plots(model, train_df, predictors, target_features):
    """
    Partial Dependence Plots (PDP) showing the marginal effect of features.
    Helps visualize if the relationship is linear, exponential, or complex.
    """
    print(f"--- Generating PDP for: {target_features} ---")
    fig, ax = plt.subplots(figsize=(12, 6))
    display = PartialDependenceDisplay.from_estimator(
        model, train_df[predictors], target_features, ax=ax
    )
    plt.show()
    return display

#---Function:permutation_importance_calc ---
def permutation_importance_calc(model, test_df, outcome, predictors, n_repeats=10):
    """
    Calculate global importance by measuring score drop after feature shuffling.
    Reliable for any model type including Stacking and Neural Networks.
    """
    X_test = test_df[predictors]
    y_test = test_df[outcome]
    
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42)
    
    importance_df = pd.DataFrame({
        'feature': predictors,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(by='importance_mean', ascending=False)
    
    print("--- Permutation Importance Summary ---")
    print(importance_df.head(10))
    return importance_df

#---Function:shap_analysis---
def shap_analysis(model, train_df, test_df, predictors):
    """
    Shapley Additive Explanations.
    Distributes the prediction value among features based on cooperative game theory.
    """
    X_test = test_df[predictors]
    
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
    except:
        explainer = shap.KernelExplainer(model.predict, shap.sample(train_df[predictors], 50))
        shap_values = explainer.shap_values(X_test)
    
    print("--- SHAP Summary Plot ---")
    shap.summary_plot(shap_values, X_test)
    return shap_values
