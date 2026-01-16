import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

#--- Function : correlation_check ---
def correlation_check(df: pd.DataFrame, columns: list | None = None, method: str = 'spearman'):
    """
    Calculate and display the correlation matrix for numeric columns only.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data.
    columns : list, optional
        List of columns to include in the correlation matrix. If None, all numeric columns are used.
    method : str, optional
        Method of correlation. Options include:
        - 'pearson'  : standard correlation coefficient, measures linear relationship.
        - 'spearman' : rank-based correlation, measures monotonic relationships, robust to outliers.
        - 'kendall'  : rank correlation (tau), measures monotonic relationships, good for small datasets.
        - 'pearsonr' : same as 'pearson', included for clarity in some cases.

    Returns:
    --------
    corr_matrix : pd.DataFrame
        Correlation matrix of the selected numeric columns.

    Example:
    --------
    correlation_check(df, columns=['score', 'age', 'height'], method='spearman')
    """
    import numpy as np

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

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

#--- Function: VIF_check ---
def VIF_check(df: pd.DataFrame, columns: list | None = None):
    """
    Calculate and display the Variance Inflation Factor (VIF) for selected columns.
    Follows best practices:
    1. Handles categorical variables (One-Hot Encoding with drop_first=True).
    2. Adds a constant (intercept) for valid R-squared calculation.
    3. Ensures numeric conversion (float64) for stability.
    4. Removes constant columns.
    """

    # Selection and copy
    df_copy = df[columns].copy() if columns is not None else df.copy()
    
    # Drop rows with missing values (VIF requirement)
    df_copy = df_copy.dropna()

    # Encode categorical variables with drop_first=True 
    # to avoid the Dummy Variable Trap
    df_encoded = pd.get_dummies(df_copy, drop_first=True, dtype=float)

    # Remove constant columns (zero variance) as they break VIF
    df_encoded = df_encoded.loc[:, df_encoded.nunique() > 1]

    # Add a constant term (intercept) - CRITICAL for statsmodels VIF
    # This ensures the model has an intercept, making R-squared calculations valid
    df_with_const = add_constant(df_encoded)

    if df_with_const.shape[1] < 2:
        raise ValueError("Not enough valid predictors to compute VIF.")

    # Compute VIF
    # We use df_with_const to calculate but exclude 'const' from final display
    vif_series = []
    for i in range(df_with_const.shape[1]):
        vif_series.append(variance_inflation_factor(df_with_const.values, i))

    vif_data = pd.DataFrame({
        'Feature': df_with_const.columns,
        'VIF': vif_series
    })

    # Final filtering and formatting
    # We remove the 'const' row as it's not a feature
    vif_data = vif_data[vif_data['Feature'] != 'const']
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
    vif_data['VIF'] = vif_data['VIF'].round(3)

    # Display VIF table
    print("\n=== Variance Inflation Factors (Optimized) ===")
    print(vif_data.to_string(index=False))
