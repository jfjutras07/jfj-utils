import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Function : multicollinearity_check ---
def multicollinearity_check(df: pd.DataFrame, method: str = 'spearman', output: str = 'both') -> pd.DataFrame | dict:
    """
    Analyze multicollinearity in a dataset using correlation matrix and VIF.
    Automatically encodes categorical variables for VIF calculation.
    """
    #Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    #Keep numeric columns only
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found after encoding.")
    
    #Drop constant columns (variance = 0)
    numeric_df = numeric_df.loc[:, numeric_df.var() != 0]
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns left after dropping zero-variance columns.")

    #Correlation matrix
    method = method.lower()
    if method not in ['spearman', 'pearson']:
        raise ValueError("Method must be 'spearman' or 'pearson'.")
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
