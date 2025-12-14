import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_multicollinearity(df, threshold=5.0, method='spearman'):
    """
    Analyze multicollinearity in a dataset using correlation matrix and VIF.
    Returns clean pandas DataFrames for Jupyter display.

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
            "correlation_matrix": pandas DataFrame of correlations,
            "vif": pandas DataFrame of features with their VIF scores,
            "high_vif_features": list of features with VIF > threshold,
            "has_multicollinearity": bool
        }
    """

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < df.shape[1]:
        raise ValueError("Input DataFrame must contain only numerical variables.")

    # Correlation matrix
    if method.lower() not in ['spearman', 'pearson']:
        raise ValueError("Method must be 'spearman' or 'pearson'.")
    corr_matrix = numeric_df.corr(method=method.lower()).round(3)

    # VIF calculation
    vif_list = [variance_inflation_factor(numeric_df.values, i) for i in range(numeric_df.shape[1])]
    vif_data = pd.DataFrame({
        'Feature': numeric_df.columns,
        'VIF': np.round(vif_list, 3)
    }).sort_values('VIF', ascending=False).reset_index(drop=True)

    # Identify high VIF features
    high_vif_features = vif_data[vif_data['VIF'] > threshold]['Feature'].tolist()
    has_multicollinearity = len(high_vif_features) > 0

    # Return clean DataFrames and lists
    return {
        'correlation_matrix': corr_matrix,
        'vif': vif_data,
        'high_vif_features': high_vif_features,
        'has_multicollinearity': has_multicollinearity
    }
