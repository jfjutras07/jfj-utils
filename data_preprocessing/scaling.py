import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

#---Function:minmax_scaler---
def minmax_scaler(train_df, test_df, columns=None):
    """
    Apply MinMaxScaler to project features into a [0, 1] range.
    Independent function: handles detection, fit on train, and transform on test.
    """
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    if columns is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in numeric_cols if not train_df[c].dropna().isin([0, 1]).all()]

    if not columns:
        return train_scaled, test_scaled, None

    scaler = MinMaxScaler()
    train_scaled[columns] = scaler.fit_transform(train_df[columns])
    test_scaled[columns] = scaler.transform(test_df[columns])

    print(f"--- MinMaxScaler Summary ---")
    print(f"Scaled: {len(columns)} features | Fit on Train only")
    print("-" * 35)

    return train_scaled, test_scaled, scaler

#---Function:robust_scaler---
def robust_scaler(train_df, test_df, columns=None):
    """
    Apply RobustScaler using the Interquartile Range (IQR).
    Independent function: best for datasets with significant outliers.
    """
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    if columns is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in numeric_cols if not train_df[c].dropna().isin([0, 1]).all()]

    if not columns:
        return train_scaled, test_scaled, None

    scaler = RobustScaler()
    train_scaled[columns] = scaler.fit_transform(train_df[columns])
    test_scaled[columns] = scaler.transform(test_df[columns])

    print(f"--- RobustScaler Summary ---")
    print(f"Scaled: {len(columns)} features | Outlier-resistant logic applied")
    print("-" * 35)

    return train_scaled, test_scaled, scaler

#---Function:standard_scaler---
def standard_scaler(train_df, test_df, columns=None):
    """
    Apply StandardScaler to achieve zero mean and unit variance.
    Independent function: standardizes continuous variables while protecting binary flags.
    """
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    if columns is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in numeric_cols if not train_df[c].dropna().isin([0, 1]).all()]

    if not columns:
        return train_scaled, test_scaled, None

    scaler = StandardScaler()
    train_scaled[columns] = scaler.fit_transform(train_df[columns])
    test_scaled[columns] = scaler.transform(test_df[columns])

    print(f"--- StandardScaler Summary ---")
    print(f"Scaled: {len(columns)} features | Zero mean/Unit variance achieved")
    print("-" * 35)

    return train_scaled, test_scaled, scaler
