import pandas as pd
from scipy.stats import skew, kurtosis

#--- Function : numeric_skew_kurt ---
def numeric_skew_kurt(df, numeric_cols):
    """
    Compute skewness and kurtosis for selected numeric columns.

    Parameters:
    - df: pandas DataFrame
    - numeric_cols: list of numeric columns to analyze (must be explicitly provided)

    Returns:
    - skew_kurt_df: DataFrame with skewness and kurtosis for each column
    """
    results = []
    
    for col in numeric_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue
            
        col_data = df[col].dropna()
        results.append({
            "Column": col,
            "Skewness": skew(col_data),
            "Kurtosis": kurtosis(col_data)
        })
    
    skew_kurt_df = pd.DataFrame(results)
    return skew_kurt_df
