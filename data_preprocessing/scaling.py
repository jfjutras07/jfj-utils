import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Optional

#--- Class : categorical_encoder ---
class categorical_encoder(BaseEstimator, TransformerMixin):
    """
    All-in-one encoder for Binary, Ordinal, and One-Hot encoding.
    """
    def __init__(self, 
                 mapping_rules: Optional[Dict[str, Dict]] = None, 
                 one_hot_cols: Optional[List[str]] = None,
                 drop_first: bool = True,
                 strict_mapping: bool = True):
        self.mapping_rules = mapping_rules if mapping_rules else {}
        self.one_hot_cols = one_hot_cols if one_hot_cols else []
        self.drop_first = drop_first
        self.strict_mapping = strict_mapping
        self.one_hot_features_ = None

    def fit(self, X: pd.DataFrame, y=None):
        # Learn One-Hot columns to ensure consistency
        if self.one_hot_cols:
            X_oh = pd.get_dummies(X[self.one_hot_cols], columns=self.one_hot_cols, drop_first=self.drop_first)
            self.one_hot_features_ = X_oh.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        #Apply Manual Mappings (Binary & Ordinal)
        for col, mapping in self.mapping_rules.items():
            if col in X.columns:
                initial_na = X[col].isna()
                X[col] = X[col].map(mapping)
                
                if self.strict_mapping:
                    invalid_mask = X[col].isna() & ~initial_na
                    if invalid_mask.any():
                        invalid_vals = X.loc[invalid_mask, col].unique()
                        raise ValueError(f"Mapping error in '{col}': {invalid_vals} not found in rules.")

        #Apply One-Hot Encoding
        if self.one_hot_cols:
            X_oh = pd.get_dummies(X, columns=self.one_hot_cols, drop_first=self.drop_first)
            
            # Identify columns that are NOT part of the new One-Hot features
            # (Essentially everything that wasn't encoded)
            other_cols = [c for c in X_oh.columns if c not in self.one_hot_features_]
            
            # Reindex One-Hot part and combine
            X_oh_part = X_oh.reindex(columns=list(other_cols) + list(self.one_hot_features_), fill_value=0)
            return X_oh_part

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
