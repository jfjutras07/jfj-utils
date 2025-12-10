import pandas as pd
from scipy.stats import skew, kurtosis

#--- Function : numeric_skew_kurt ---
def numeric_skew_kurt(df, numeric_cols=None):
    """
    Compute skewness and kurtosis for numeric columns.

    Parameters:
    - df: pandas DataFrame
    - numeric_cols: list of numeric columns to analyze (default: all numeric columns)

    Returns:
    - skew_kurt_df: DataFrame with skewness and kurtosis
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    results = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        results.append({
            "Column": col,
            "Skewness": skew(col_data),
            "Kurtosis": kurtosis(col_data)
        })
    
    skew_kurt_df = pd.DataFrame(results)
    return skew_kurt_df

