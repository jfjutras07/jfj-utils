import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

#--- Function : check_multicollinearity ---
def check_multicollinearity(df, threshold=5.0):
    """
    Analyze multicollinearity in a dataset using correlation matrix and VIF.
    Designed for regression workflows to detect unstable or redundant predictors.

    Parameters
    ----------
    df : DataFrame
        Dataset containing only numerical features.
    threshold : float, optional
        VIF threshold above which a feature is considered problematic.
        Common values: 5.0 (moderate), 10.0 (severe).

    Returns
    -------
    dict
        {
            "correlation_matrix": DataFrame,
            "vif": DataFrame with VIF scores,
            "high_vif_features": list of feature names,
            "has_multicollinearity": bool
        }
    """

    #Ensure only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < df.shape[1]:
        raise ValueError("Input DataFrame must contain only numerical variables.")

    #Correlation Matrix
    corr_matrix = numeric_df.corr()

    #VIF Calculation
    vif_data = []
    for i in range(numeric_df.shape[1]):
        vif_data.append({
            "feature": numeric_df.columns[i],
            "vif": variance_inflation_factor(numeric_df.values, i)
        })

    vif_df = pd.DataFrame(vif_data)

    #Detect features exceeding the threshold
    high_vif = vif_df[vif_df["vif"] > threshold]["feature"].tolist()

    return {
        "correlation_matrix": corr_matrix,
        "vif": vif_df,
        "high_vif_features": high_vif,
        "has_multicollinearity": len(high_vif) > 0
    }
