import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

#--- Function : multicollinearity_check ---
def multicollinearity_check(
    df: pd.DataFrame, 
    method: str = 'spearman', 
    output: str = 'both'  # options: 'both', 'correlation', 'vif'
) -> pd.DataFrame | dict:
    """
    Analyze multicollinearity in a dataset using correlation matrix and VIF.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing numeric features.
    method : str, optional
        Correlation method: 'spearman' or 'pearson'. Default is 'spearman'.
    output : str, optional
        Which result to return: 'both', 'correlation', or 'vif'. Default is 'both'.

    Returns
    -------
    pd.DataFrame or dict
        Depending on `output`:
        - 'correlation' -> correlation matrix
        - 'vif' -> VIF DataFrame
        - 'both' -> dict with both correlation matrix and VIF
    """
    #Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found in DataFrame.")
    if (numeric_df.var() == 0).any():
        raise ValueError("Some columns have zero variance, VIF cannot be computed.")

    #Validate method
    method = method.lower()
    if method not in ['spearman', 'pearson']:
        raise ValueError("Method must be 'spearman' or 'pearson'.")

    #Correlation matrix
    corr_matrix = numeric_df.corr(method=method).round(3)

    #VIF calculation
    vif_data = pd.DataFrame({
        'Feature': numeric_df.columns,
        'VIF': [variance_inflation_factor(numeric_df.values, i) 
                for i in range(numeric_df.shape[1])]
    }).sort_values('VIF', ascending=False).reset_index(drop=True)
    vif_data['VIF'] = vif_data['VIF'].round(3)

    #Return based on output parameter
    if output == 'correlation':
        return corr_matrix
    elif output == 'vif':
        return vif_data
    elif output == 'both':
        return {'correlation_matrix': corr_matrix, 'vif': vif_data}
    else:
        raise ValueError("Output parameter must be 'both', 'correlation', or 'vif'.")
