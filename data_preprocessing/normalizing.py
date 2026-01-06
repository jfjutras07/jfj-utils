import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Union, Tuple

#--- Function : normalize_columns ---
def normalize_columns(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    method: str = "standard"
) -> Tuple[Union[pd.DataFrame, List[pd.DataFrame]], Union[object, List[object]]]:
    """
    Automatically normalize all numeric or ordinal columns in one or multiple datasets.

    Parameters:
        dfs : pd.DataFrame or list of pd.DataFrame
            Dataset(s) containing numeric/ordinal columns to normalize.
        method : str, default="standard"
            Normalization method. Options:
            - "standard": StandardScaler (mean=0, std=1)
            - "minmax": MinMaxScaler (scaled to 0-1)
            - "robust": RobustScaler (median & IQR, less sensitive to outliers)

    Returns:
        Tuple of (normalized datasets, scaler(s)):
            - normalized dataset(s)
            - fitted scaler object(s)
    """
    #Ensure dfs is iterable
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    #Select scaler
    if method.lower() == "standard":
        ScalerClass = StandardScaler
    elif method.lower() == "minmax":
        ScalerClass = MinMaxScaler
    elif method.lower() == "robust":
        ScalerClass = RobustScaler
    else:
        raise ValueError(f"Unknown normalization method '{method}'. Choose 'standard', 'minmax', or 'robust'.")

    normalized_dfs = []
    scalers = []

    for df_idx, df in enumerate(dfs):
        df_copy = df.copy()

        # Automatically select numeric/ordinal columns
        numeric_cols = df_copy.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if not numeric_cols:
            print(f"WARNING: No numeric/ordinal columns detected in dataset {df_idx}. Skipping normalization.")
            normalized_dfs.append(df_copy)
            scalers.append(None)
            continue

        # Apply scaler
        scaler = ScalerClass()
        df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
        normalized_dfs.append(df_copy)
        scalers.append(scaler)

    #Return single objects if single dataset
    if single_df:
        return normalized_dfs[0], scalers[0]
    else:
        return normalized_dfs, scalers
