import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

#--- Function : correlation_check ---
def correlation_check(df: pd.DataFrame, columns: list | None = None, method: str = 'spearman'):
    """
    Calculate and display the correlation matrix for numeric columns only.
    """
    #Select only columns if provided
    if columns is not None:
        df = df[columns]

    #Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation.")

    #Calculate correlation matrix
    corr_matrix = numeric_df.corr(method=method.lower()).round(3)

    #Display correlation matrix
    print("\n=== Correlation Matrix ===")
    print(corr_matrix.to_string())

#--- Function : VIF_check ---
def VIF_check(df: pd.DataFrame, columns: list | None = None):
    """
    Calculate and display the Variance Inflation Factor (VIF) for selected columns.
    Automatically encodes categorical variables and handles constants and non-numeric columns.
    """
    #Copy only selected columns to avoid modifying original df
    df_copy = df[columns].copy() if columns is not None else df.copy()
    
    #Encode categorical variables with drop_first=True to avoid singularity
    df_encoded = pd.get_dummies(df_copy, drop_first=True)
    
    #Force numeric and replace non-finite values
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    
    #Replace NaN with 0 to keep all columns
    df_encoded = df_encoded.fillna(0)
    
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

    #Display VIF table
    print("\n=== VIF Table ===")
    print(vif_data.to_string())
