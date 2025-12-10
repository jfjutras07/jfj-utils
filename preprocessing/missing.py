import pandas as pd

#--- Function : missing_summary ---
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary table of missing values in a DataFrame.
    
    Columns:
        - missing_count: number of missing values per column
        - missing_percent: percentage of missing values per column
    Sorted by missing_count descending.
    """
    missing_count = df.isnull().sum()
    missing_percent = 100 * missing_count / len(df)
    
    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent
    }).sort_values(by="missing_count", ascending=False)
    
    return summary

#--- Function : missing_stats ---
def missing_stats(df: pd.DataFrame) -> dict:
    """
    Returns general statistics about missing values in a DataFrame:
        - total_missing: total number of missing values
        - percent_missing: percent of missing values in the entire DataFrame
        - num_columns_missing: number of columns with at least one missing value
        - num_rows_missing: number of rows with at least one missing value
    """
    total_missing = df.isnull().sum().sum()
    percent_missing = 100 * total_missing / (df.shape[0] * df.shape[1])
    num_columns_missing = df.isnull().any().sum()
    num_rows_missing = df.isnull().any(axis=1).sum()
    
    return {
        "total_missing": total_missing,
        "percent_missing": percent_missing,
        "num_columns_missing": num_columns_missing,
        "num_rows_missing": num_rows_missing
    }
