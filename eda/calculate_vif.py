import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df, features):
    """
    Calculate and display Variance Inflation Factor (VIF) for each feature.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing predictor variables.
    features : list of str
        List of column names to include in VIF calculation.

    Returns:
    --------
    vif_df : pd.DataFrame
        DataFrame with VIF values for each predictor.
    """
    #Extract the predictor matrix
    X = df[features].copy()
    
    #If any categorical variable, convert to dummies
    X = pd.get_dummies(X, drop_first=True)
    
    vif_data = []
    for i in range(X.shape[1]):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({'Variable': X.columns[i], 'VIF': vif})
    
    vif_df = pd.DataFrame(vif_data).sort_values(by='VIF', ascending=False).reset_index(drop=True)
    print(vif_df)
