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
    Automatically encodes categorical variables, removes constant columns, and handles numeric conversion.
    """

    #Copy selected columns to avoid modifying original DataFrame
    df_copy = df[columns].copy() if columns is not None else df.copy()
    
    #Drop rows with missing values
    df_copy = df_copy.dropna()

    #Encode categorical variables
    df_encoded = pd.get_dummies(df_copy, drop_first=True)

    #Convert all columns to float
    df_encoded = df_encoded.astype(float)

    #Remove constant columns
    df_encoded = df_encoded.loc[:, df_encoded.var() > 0]

    if df_encoded.shape[1] < 2:
        raise ValueError("Not enough valid predictors to compute VIF.")

    #Compute VIF
    vif_data = pd.DataFrame({
        'Feature': df_encoded.columns,
        'VIF': [variance_inflation_factor(df_encoded.values, i) for i in range(df_encoded.shape[1])]
    }).sort_values('VIF', ascending=False).reset_index(drop=True)

    #Round for readability
    vif_data['VIF'] = vif_data['VIF'].round(3)

    #Display VIF table
    print("\n=== VIF DES PRÉDICTEURS ===")
    print(vif_data.to_string(index=False))

    return vif_data
