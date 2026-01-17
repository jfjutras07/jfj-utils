import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Union, Optional

#--- Class : feature_scaler ---
class feature_scaler(BaseEstimator, TransformerMixin):
    """
    Generic numerical scaler supporting multiple methods.
    Maintains DataFrame structure and column names.
    If columns is None, it scales all numeric features received.
    """
    def __init__(self, columns: Optional[List[str]] = None, method: str = 'robust'):
        self.columns = columns
        self.method = method
        self.scaler_ = None
        
        self._method_map = {
            'robust': RobustScaler,
            'standard': StandardScaler,
            'minmax': MinMaxScaler
        }
        
        if self.method not in self._method_map:
            raise ValueError(
                f"Method '{self.method}' not supported. "
                f"Choose from: {list(self._method_map.keys())}"
            )

    def fit(self, X: pd.DataFrame, y=None):
        self.target_cols_ = self.columns if self.columns else X.columns.tolist()
        
        if self.target_cols_:
            existing_cols = [c for c in self.target_cols_ if c in X.columns]
            if existing_cols:
                self.scaler_ = self._method_map[self.method]()
                self.scaler_.fit(X[existing_cols])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.scaler_ and hasattr(self, 'target_cols_'):
            existing_cols = [c for c in self.target_cols_ if c in X.columns]
            if existing_cols:
                X[existing_cols] = self.scaler_.transform(X[existing_cols])
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.target_cols_)
        return np.array(input_features)

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
