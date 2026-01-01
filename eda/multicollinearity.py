import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

#--- Function : correlation_check ---
def correlation_check(df: pd.DataFrame, columns: list | None = None, method: str = 'spearman') -> pd.DataFrame:
    """
    Calculate the correlation matrix for selected columns.
    Automatically encodes categorical variables.
    """
    if columns is not None:
        df = df[columns]

    # Encode categorical columns
    df_encoded = pd.get_dummies(df, drop_first=False)

    # Keep only numeric columns (after encoding)
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation.")

    corr_matrix = numeric_df.corr(method=method.lower()).round(3)
    return corr_matrix

#--- Function : VIF_check ---
def VIF_check(df: pd.DataFrame, columns: list | None = None) -> pd.DataFrame:
    """
    Calculate the Variance Inflation Factor (VIF) for selected columns.
    Automatically encodes categorical variables and handles constants.
    """
    if columns is not None:
        df = df[columns]

    #Encode categorical variables automatically
    df_encoded = pd.get_dummies(df, drop_first=False)

    #Force all columns to numeric
    df_encoded = df_encoded.apply(pd.to_numeric)

    #Remove constant columns
    df_encoded = df_encoded.loc[:, df_encoded.var() != 0]

    if df_encoded.shape[1] == 0:
        raise ValueError("No valid columns left for VIF calculation.")

    #Calculate VIF
    vif_data = pd.DataFrame({
        'Feature': df_encoded.columns,
        'VIF': [variance_inflation_factor(df_encoded.values, i) for i in range(df_encoded.shape[1])]
    }).sort_values('VIF', ascending=False).reset_index(drop=True)

    #Round for readability
    vif_data['VIF'] = vif_data['VIF'].round(3)

    return vif_data

