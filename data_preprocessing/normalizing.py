import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Union, Tuple

#--- Function : normalize_columns ---
def normalize_columns(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    method: str = "standard"
) -> Tuple[Union[pd.DataFrame, List[pd.DataFrame]], Union[object, List[object]]]:
    """
    Normalize only continuous and ordinal numeric columns in one or multiple datasets.
    Binary and one-hot encoded columns are automatically excluded.

    Parameters:
        dfs : pd.DataFrame or list of pd.DataFrame
        method : str, default="standard" ("standard", "minmax", "robust")

    Returns:
        Tuple of (normalized datasets, scaler(s))
    """
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    # Select scaler
    if method.lower() == "standard":
        ScalerClass = StandardScaler
    elif method.lower() == "minmax":
        ScalerClass = MinMaxScaler
    elif method.lower() == "robust":
        ScalerClass = RobustScaler
    else:
        raise ValueError(f"Unknown normalization method '{method}'")

    normalized_dfs = []
    scalers = []

    for df_idx, df in enumerate(dfs):
        df_copy = df.copy()

        #Detect numeric columns
        numeric_cols = df_copy.select_dtypes(include=["int64", "float64"]).columns.tolist()

        #Exclude binary columns (0/1) and one-hot dummies
        cols_to_scale = [c for c in numeric_cols if df_copy[c].nunique() > 2]

        if not cols_to_scale:
            print(f"WARNING: No continuous/ordinal numeric columns detected in dataset {df_idx}. Skipping normalization.")
            normalized_dfs.append(df_copy)
            scalers.append(None)
            continue

        #Apply scaler
        scaler = ScalerClass()
        df_copy[cols_to_scale] = scaler.fit_transform(df_copy[cols_to_scale])
        normalized_dfs.append(df_copy)
        scalers.append(scaler)

    if single_df:
        return normalized_dfs[0], scalers[0]
    else:
        return normalized_dfs, scalers
