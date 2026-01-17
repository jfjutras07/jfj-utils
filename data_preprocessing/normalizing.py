import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Union, Tuple
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# --- Class : skewness_corrector ---
class skewness_corrector(BaseEstimator, TransformerMixin):
    """
    Applies log1p transformation to numeric columns.
    You can either provide a manual list of columns, or let it detect skewed columns automatically.
    """
    def __init__(self, threshold=0.75, manual_cols=None, epsilon=1e-6):
        self.threshold = threshold
        self.cols_to_transform_ = []
        self.manual_cols = manual_cols
        self.epsilon = epsilon  # small value to avoid log(0)

    def fit(self, X, y=None):
        numeric_df = X.select_dtypes(include=[np.number])
        if self.manual_cols is not None:
            # Use only manual columns
            self.cols_to_transform_ = [col for col in self.manual_cols if col in numeric_df.columns]
        else:
            # Auto-detect skewed columns
            skewness = numeric_df.skew()
            self.cols_to_transform_ = skewness[abs(skewness) > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in self.cols_to_transform_:
            if col in numeric_cols:
                min_val = X[col].min()
                if min_val <= 0:
                    X[col] = X[col] - min_val + self.epsilon
                X[col] = np.log1p(X[col])
        return X

#--- Function : normalize_columns ---
def normalize_columns(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    method: str = "standard"
) -> Tuple[Union[pd.DataFrame, List[pd.DataFrame]], Union[object, List[object]]]:
    """
    Normalize only continuous and ordinal numeric columns in one or multiple datasets.
    Binary and one-hot encoded columns are automatically excluded.
    The scaler is fitted on the first dataset and applied to all subsequent ones
    to prevent data leakage.

    Parameters:
        dfs : pd.DataFrame or list of pd.DataFrame
        method : str, default="standard" ("standard", "minmax", "robust")

    Returns:
        Tuple of (normalized datasets, scaler)
    """
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    # Select scaler class
    if method.lower() == "standard":
        scaler = StandardScaler()
    elif method.lower() == "minmax":
        scaler = MinMaxScaler()
    elif method.lower() == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method '{method}'")

    # 1. Identify columns to scale using the FIRST dataset
    first_df = dfs[0]
    numeric_cols = first_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Logic: More than 2 unique values to exclude binary/dummies
    cols_to_scale = [c for c in numeric_cols if first_df[c].nunique() > 2]

    if not cols_to_scale:
        print(f"WARNING: No continuous/ordinal numeric columns detected. Skipping normalization.")
        return (dfs[0] if single_df else dfs), None

    # 2. Fit the scaler on the first dataset only (Train)
    scaler.fit(first_df[cols_to_scale])

    # 3. Transform all datasets using the fitted scaler
    normalized_dfs = []
    for df in dfs:
        df_copy = df.copy()
        # Force conversion to float64 to accommodate scaled values
        df_copy[cols_to_scale] = scaler.transform(df_copy[cols_to_scale].astype("float64"))
        normalized_dfs.append(df_copy)

    print(
        f"Normalization ({method}) successfully applied to {len(cols_to_scale)} columns "
        f"on {len(normalized_dfs)} dataset(s)."
    )

    if single_df:
        return normalized_dfs[0], scaler
    else:
        return normalized_dfs, scaler
