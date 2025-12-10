import pandas as pd
from scipy.stats.mstats import winsorize
from scipy.stats import zscore

#--- Function : detect_outliers_zscore ---
def detect_outliers_zscore(df: pd.DataFrame, columns: list, threshold: float = 3.0) -> dict:
    """
    Detect outliers using z-score method (works well with normal distributions)
    Returns a dictionary with column names and row indices of outliers.
    """
    outliers = {}
    for col in columns:
        if col in df.columns:
            z_scores = zscore(df[col].dropna())
            outlier_idx = df[col].iloc[z_scores > threshold].index.tolist()
            outliers[col] = outlier_idx
    return outliers

#--- Function : detect_outliers_iqr ---
def detect_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> dict:
    """
    Detect outliers in numerical columns using the IQR method (works well with non-normal or skewed distributions)
    Returns a dictionary with column names as keys and row indices of outliers as values.
    """
    outliers = {}
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            outlier_idx = df[(df[col] < lower) | (df[col] > upper)].index.tolist()
            outliers[col] = outlier_idx
    return outliers

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

