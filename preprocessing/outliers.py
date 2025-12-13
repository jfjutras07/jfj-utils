import pandas as pd
from scipy.stats.mstats import winsorize
from scipy.stats import zscore

#--- Function : detect_outliers_zscore ---
def detect_outliers_zscore(df: pd.DataFrame, columns: list, threshold: float = 3.0) -> dict:
    """
    Count outliers using z-score method (works well with normal distributions).
    Returns a dictionary with:
      - number of outliers per column
      - total number of outliers across all columns
    """
    outlier_counts = {}
    total_outliers = 0
    
    for col in columns:
        if col in df.columns:
            z_scores = zscore(df[col].dropna())
            outlier_count = (z_scores > threshold).sum()
            outlier_counts[col] = outlier_count
            total_outliers += outlier_count
    
    outlier_counts['Total_outliers'] = total_outliers
    return outlier_counts

#--- Function : detect_outliers_iqr ---
def detect_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> dict:
    """
    Count the number of outliers in numerical columns using the IQR method.
    Returns a dictionary with:
      - number of outliers per column
      - total number of outliers across all columns (key 'Total_outliers')
    """
    outlier_counts = {}
    total_outliers = 0
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            outlier_count = df[(df[col] < lower) | (df[col] > upper)].shape[0]
            outlier_counts[col] = outlier_count
            total_outliers += outlier_count
    
    outlier_counts['Total_outliers'] = total_outliers
    return outlier_counts

#--- Function : winsorize_columns ---
def winsorize_columns(df: pd.DataFrame, columns: list, limits: tuple = (0.05, 0.05)) -> pd.DataFrame:
    """
    Apply winsorization to specified numerical columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of numerical columns to winsorize
        limits (tuple): Tuple of (lower_limit, upper_limit) in proportion (0-1)
                        e.g., (0.05, 0.05) winsorizes the bottom 5% and top 5%
    
    Returns:
        pd.DataFrame: Copy of DataFrame with winsorized columns
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            try:
                #Apply winsorization and convert back to Series
                df_copy[col] = pd.Series(winsorize(df_copy[col], limits=limits), index=df_copy.index)
            except Exception as e:
                print(f"Warning: Could not winsorize column {col}: {e}")
        else:
            print(f"Warning: Column {col} not found in DataFrame")
    
    return df_copy

