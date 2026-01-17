import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Union, Optional, Callable

#--- Class : group_imputer ---
class group_imputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using group-level statistics learned from training data.
    """
    def __init__(self, group_col: str, target_col: str, strategy: str = 'median'):
        self.group_col = group_col
        self.target_col = target_col
        self.strategy = strategy
        self.fill_values = None
        self.global_fallback = None

    def fit(self, X, y=None):
        # Ensure we are working with a DataFrame
        if not hasattr(X, 'groupby'):
            X = pd.DataFrame(X)

        if self.strategy == 'median':
            self.fill_values = X.groupby(self.group_col)[self.target_col].median()
            self.global_fallback = X[self.target_col].median()
        elif self.strategy == 'mean':
            self.fill_values = X.groupby(self.group_col)[self.target_col].mean()
            self.global_fallback = X[self.target_col].mean()
        elif self.strategy == 'mode':
            self.fill_values = X.groupby(self.group_col)[self.target_col].apply(
                lambda x: x.mode()[0] if not x.mode().empty else np.nan
            )
            self.global_fallback = X[self.target_col].mode()[0]
        else:
            raise ValueError(f"Strategy '{self.strategy}' not recognized. Use 'mean', 'median' or 'mode'.")
            
        return self

    def transform(self, X):
        X = X.copy()
        # Group-level fill
        X[self.target_col] = X[self.target_col].fillna(X[self.group_col].map(self.fill_values))
        # Global-level fallback for unseen categories or empty groups
        X[self.target_col] = X[self.target_col].fillna(self.global_fallback)
        return X

# --- Class : logical_imputer ---
class logical_imputer(BaseEstimator, TransformerMixin):
    """
    Generic transformer to apply deductive logic. 
    Example: If Experience < 1, set NumCompanies to 0.
    """
    def __init__(self, target_col: str, condition_func: Callable, fill_value=None, fill_from_col: str = None):
        self.target_col = target_col
        self.condition_func = condition_func
        self.fill_value = fill_value
        self.fill_from_col = fill_from_col

    def fit(self, X, y=None):
        # Validation: check if required columns exist during fit
        cols_to_check = [self.target_col]
        if self.fill_from_col:
            cols_to_check.append(self.fill_from_col)
            
        missing_cols = [c for c in cols_to_check if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame.")
        return self

    def transform(self, X):
        # Use copy only to avoid SettingWithCopyWarning on the original df
        X = X.copy()
        
        # Apply the logic: condition must be met AND target must be NaN
        mask = self.condition_func(X) & X[self.target_col].isna()
        
        if self.fill_from_col:
            X.loc[mask, self.target_col] = X.loc[mask, self.fill_from_col]
        elif self.fill_value is not None:
            X.loc[mask, self.target_col] = self.fill_value
            
        return X
        
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
