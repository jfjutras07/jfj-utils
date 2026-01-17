import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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
