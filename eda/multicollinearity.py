import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

#--- Function : multicollinearity_check ---
def multicollinearity_check(df, method='spearman'):
    """
    Analyze multicollinearity in a dataset using correlation matrix and VIF.
   
    Parameters
    ----------
    df : DataFrame
        Dataset containing numeric features.
    method : str, optional
        Correlation method: 'spearman' or 'pearson'. Default is 'spearman'.

    Returns
    -------
    dict
        {
            "correlation_matrix": DataFrame,
            "vif": DataFrame
        }
    """

    #Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < df.shape[1]:
        raise ValueError("Input DataFrame must contain only numerical variables.")

    #Correlation matrix
    if method.lower() not in ['spearman', 'pearson']:
        raise ValueError("Method must be 'spearman' or 'pearson'.")
    corr_matrix = numeric_df.corr(method=method.lower()).round(3)

    #VIF calculation
    vif_data = pd.DataFrame({
        'Feature': numeric_df.columns,
        'VIF': [variance_inflation_factor(numeric_df.values, i) 
                for i in range(numeric_df.shape[1])]
    }).sort_values('VIF', ascending=False).reset_index(drop=True)
    vif_data['VIF'] = vif_data['VIF'].round(3)

    return {
        'correlation_matrix': corr_matrix,
        'vif': vif_data
    }
