import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

#--- Function : check_multicollinearity ---
def check_multicollinearity(df, threshold=5.0, method='spearman'):
    """
    Analyze multicollinearity in a dataset using correlation matrix and VIF.

    Parameters
    ----------
    df : DataFrame
        Dataset containing numerical features.
    threshold : float, optional
        VIF threshold above which a feature is considered problematic.
    method : str, optional
        Correlation method: 'spearman' or 'pearson'. Default is 'spearman'.

    Returns
    -------
    dict
        {
            "correlation_matrix": nicely spaced correlation table,
            "vif": nicely spaced VIF table,
            "high_vif_features": list of feature names with VIF > threshold,
            "has_multicollinearity": bool
        }
    """

    #Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < df.shape[1]:
        raise ValueError("Input DataFrame must contain only numerical variables.")

    #Compute correlation matrix
    if method.lower() not in ['spearman', 'pearson']:
        raise ValueError("Method must be 'spearman' or 'pearson'.")
    corr_matrix = numeric_df.corr(method=method.lower()).round(3)

    #VIF calculation
    vif_data = pd.DataFrame({
        "feature": numeric_df.columns,
        "vif": [variance_inflation_factor(numeric_df.values, i)
                for i in range(numeric_df.shape[1])]
    }).sort_values("vif", ascending=False).reset_index(drop=True)
    vif_data["vif"] = vif_data["vif"].round(3)

    #Identify high VIF features
    high_vif_features = vif_data[vif_data['vif'] > threshold]['feature'].tolist()
    has_multicollinearity = len(high_vif_features) > 0

    return {
        "correlation_matrix": "\n" + str(corr_matrix) + "\n",
        "vif": "\n" + str(vif_data) + "\n",
        "high_vif_features": "\n" + str(high_vif_features) + "\n",
        "has_multicollinearity": has_multicollinearity
    }
