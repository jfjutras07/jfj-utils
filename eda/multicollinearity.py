import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

#--- Function : correlation_check ---
def correlation_check(df: pd.DataFrame, columns: list | None = None, method: str = 'spearman') -> pd.DataFrame:
    """
    Calculate the correlation matrix for selected columns using the specified method.

    Parameters
    ----------
    df : DataFrame
        Dataset containing numeric and/or categorical features.
    columns : list, optional
        List of columns to include. Default is all columns.
    method : str, optional
        Correlation method: 'pearson', 'spearman', 'kendall', etc. Default is 'spearman'.

    Returns
    -------
    DataFrame
        Correlation matrix.
    """
    if columns is not None:
        df = df[columns]

    #Keep numeric columns only for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation.")

    method = method.lower()
    corr_matrix = numeric_df.corr(method=method).round(3)
    return corr_matrix

#--- Function : VIF_check ---
def VIF_check(df: pd.DataFrame, columns: list | None = None) -> pd.DataFrame:
    """
    Calculate the Variance Inflation Factor (VIF) for selected columns.
    Automatically encodes categorical variables.

    Parameters
    ----------
    df : DataFrame
        Dataset containing numeric and/or categorical features.
    columns : list, optional
        List of columns to include. Default is all columns.

    Returns
    -------
    DataFrame
        Table with features and their VIF values.
    """
    if columns is not None:
        df = df[columns]

    #Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=False)

    #Remove constant columns
    df_encoded = df_encoded.loc[:, df_encoded.var() != 0]
    if df_encoded.shape[1] == 0:
        raise ValueError("No valid columns left for VIF calculation.")

    #Compute VIF
    vif_data = pd.DataFrame({
        'Feature': df_encoded.columns,
        'VIF': [variance_inflation_factor(df_encoded.values, i) 
                for i in range(df_encoded.shape[1])]
    }).sort_values('VIF', ascending=False).reset_index(drop=True)
    vif_data['VIF'] = vif_data['VIF'].round(3)

    return vif_data
