import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Union

#--- Class : feature_scaler ---
class feature_scaler(BaseEstimator, TransformerMixin):
    """
    Generic numerical scaler supporting multiple methods.
    Maintains DataFrame structure and column names.
    """
    def __init__(self, columns: List[str], method: str = 'robust'):
        self.columns = columns
        self.method = method
        self.scaler_ = None
        
        # Mapping methods to Scikit-Learn classes
        self._method_map = {
            'robust': RobustScaler,
            'standard': StandardScaler,
            'minmax': MinMaxScaler
        }
        
        if self.method not in self._method_map:
            raise ValueError(f"Method '{self.method}' not supported. Choose from: {list(self._method_map.keys())}")

    def fit(self, X: pd.DataFrame, y=None):
        # Initialize and fit the chosen scaler
        if self.columns:
            # We filter columns that actually exist to avoid crashes
            existing_cols = [c for c in self.columns if c in X.columns]
            self.scaler_ = self._method_map[self.method]()
            self.scaler_.fit(X[existing_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.scaler_ and self.columns:
            existing_cols = [c for c in self.columns if c in X.columns]
            # Transformation and replacement in the DataFrame
            X[existing_cols] = self.scaler_.transform(X[existing_cols])
        return X

#---Function:minmax_scaler---
def minmax_scaler(train_df, test_df=None, columns=None):
    """
    Apply MinMaxScaler to project features into a [0, 1] range.
    Flexible: Handles single DF (clustering) or Train/Test split.
    """
    train_scaled = train_df.copy()
    test_scaled = test_df.copy() if test_df is not None else None

    if columns is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in numeric_cols if not train_df[c].dropna().isin([0, 1]).all()]

    if not columns:
        return (train_scaled, test_scaled, None) if test_df is not None else (train_scaled, None)

    scaler = MinMaxScaler()
    train_scaled[columns] = scaler.fit_transform(train_df[columns])
    if test_scaled is not None:
        test_scaled[columns] = scaler.transform(test_df[columns])

    print(f"--- MinMaxScaler Summary ---")
    print(f"Scaled: {len(columns)} features | Mode: {'Train/Test' if test_df is not None else 'Single DF'}")
    print("-" * 35)

    return (train_scaled, test_scaled, scaler) if test_df is not None else (train_scaled, scaler)

#---Function:robust_scaler---
def robust_scaler(train_df, test_df=None, columns=None):
    """
    Apply RobustScaler using the Interquartile Range (IQR).
    Best for datasets with significant outliers. Flexible for single or split DFs.
    """
    train_scaled = train_df.copy()
    test_scaled = test_df.copy() if test_df is not None else None

    if columns is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in numeric_cols if not train_df[c].dropna().isin([0, 1]).all()]

    if not columns:
        return (train_scaled, test_scaled, None) if test_df is not None else (train_scaled, None)

    scaler = RobustScaler()
    train_scaled[columns] = scaler.fit_transform(train_df[columns])
    if test_scaled is not None:
        test_scaled[columns] = scaler.transform(test_df[columns])

    print(f"--- RobustScaler Summary ---")
    print(f"Scaled: {len(columns)} features | Outlier-resistant logic applied")
    print("-" * 35)

    return (train_scaled, test_scaled, scaler) if test_df is not None else (train_scaled, scaler)

#---Function:standard_scaler---
def standard_scaler(train_df, test_df=None, columns=None):
    """
    Apply StandardScaler to achieve zero mean and unit variance.
    Standardizes continuous variables while protecting binary flags.
    """
    train_scaled = train_df.copy()
    test_scaled = test_df.copy() if test_df is not None else None

    if columns is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in numeric_cols if not train_df[c].dropna().isin([0, 1]).all()]

    if not columns:
        return (train_scaled, test_scaled, None) if test_df is not None else (train_scaled, None)

    scaler = StandardScaler()
    train_scaled[columns] = scaler.fit_transform(train_df[columns])
    if test_scaled is not None:
        test_scaled[columns] = scaler.transform(test_df[columns])

    print(f"--- StandardScaler Summary ---")
    print(f"Scaled: {len(columns)} features | Zero mean/Unit variance achieved")
    print("-" * 35)

    return (train_scaled, test_scaled, scaler) if test_df is not None else (train_scaled, scaler)
