import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

#--- Function : check_model_stability ---
def check_model_stability(model, X, y, cv=5, scoring='r2'):
    """
    Algorithmic Stability Check.
    Measures the variance of model performance across multiple folds to 
    validate consistency and detect overfitting.

    Parameters:
    -----------
    model : fitted estimator or Pipeline
        The champion model or pipeline to evaluate.
    X : pd.DataFrame or np.array
        Feature matrix.
    y : pd.Series or np.array
        Target variable.
    cv : int
        Number of folds for cross-validation.
    scoring : str
        Metric to evaluate (e.g., 'r2', 'neg_mean_absolute_error').
    """
    
    #Execution of Cross-Validation
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    #Standardizing negative metrics (MAE/RMSE) for display
    if scoring.startswith('neg_'):
        scores = -scores
        metric_label = scoring.replace('neg_', '').upper()
    else:
        metric_label = scoring.upper()

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    #Coefficient of Variation (Stability Ratio)
    #A low ratio indicates a robust model that generalizes well.
    stability_ratio = (std_score / mean_score) if mean_score != 0 else 0

    print(f"--- Algorithmic Stability Check ({metric_label}) ---")
    print(f"Number of Folds : {cv}")
    print(f"Scores per Fold : {np.round(scores, 4)}")
    print(f"Mean Score      : {mean_score:.4f}")
    print(f"Std Deviation   : {std_score:.4f}")
    print(f"Stability Ratio : {stability_ratio:.2%}")
    print("-" * 40)

    #Formal Diagnostic
    if stability_ratio > 0.15:
        print("Status: UNSTABLE. High variance detected between folds.")
    elif stability_ratio > 0.05:
        print("Status: CAUTION. Moderate variance observed.")
    else:
        print("Status: STABLE. Model performance is consistent.")
    
    print("-" * 40)
    
    return {
        "mean": mean_score,
        "std": std_score,
        "stability_ratio": stability_ratio
    }
