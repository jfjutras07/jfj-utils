import pandas as pd
from typing import Dict, List, Union

#--- Function : binary_encode_columns ---
def binary_encode_columns(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    binary_mappings: Dict[str, Dict],
    strict: bool = True,
    train_reference: pd.DataFrame | None = None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Apply explicit binary encoding to selected columns for one or multiple datasets
    and optionally use a train reference to avoid data leakage.
    """
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    #If train_reference is provided, we map using its values
    ref_mapping = {}
    if train_reference is not None:
        for col, mapping in binary_mappings.items():
            if col not in train_reference.columns:
                raise KeyError(f"Column '{col}' not found in train_reference")
            ref_mapping[col] = mapping

    encoded_dfs = []
    for df_idx, df in enumerate(dfs):
        df = df.copy()
        for col, mapping in (ref_mapping if ref_mapping else binary_mappings).items():
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in dataset {df_idx}")
            df[col] = df[col].map(mapping)
            invalid_mask = ~df[col].isin([0,1]) & df[col].notna()
            if invalid_mask.any():
                invalid_values = df.loc[invalid_mask, col].unique()
                message = (
                    f"Invalid values found after binary encoding in column '{col}' "
                    f"(dataset {df_idx}): {invalid_values}"
                )
                if strict:
                    raise ValueError(message)
                else:
                    print(f"WARNING: {message}")
        encoded_dfs.append(df)

    print(
        f"Binary encoding successfully applied to "
        f"{len(binary_mappings)} columns on {len(encoded_dfs)} dataset(s)."
    )
    return encoded_dfs[0] if single_df else encoded_dfs

#--- Function : label_encode_columns ---
def label_encode_columns(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    categorical_cols: List[str],
    train_reference: pd.DataFrame | None = None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Apply label encoding to selected categorical columns for one or multiple datasets.
    If train_reference is provided, use the factorization from train_reference.
    """
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    #Compute reference mapping if train_reference provided
    ref_mappings = {}
    if train_reference is not None:
        for col in categorical_cols:
            if col not in train_reference.columns:
                raise KeyError(f"Column '{col}' not found in train_reference")
            codes, uniques = pd.factorize(train_reference[col], sort=True)
            ref_mappings[col] = {cat: i for i, cat in enumerate(uniques)}

    encoded_dfs = []
    for df_idx, df in enumerate(dfs):
        df = df.copy()
        for col in categorical_cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in dataset {df_idx}")
            mapping = ref_mappings[col] if ref_mappings else None
            if mapping:
                df[col] = df[col].map(mapping)
            else:
                df[col], _ = pd.factorize(df[col], sort=True)
            df[col] = df[col].astype(int)
        encoded_dfs.append(df)

    print(
        f"Label encoding successfully applied to {len(categorical_cols)} columns "
        f"on {len(encoded_dfs)} dataset(s)."
    )
    return encoded_dfs[0] if single_df else encoded_dfs

#--- Function : one_hot_encode_columns ---
def one_hot_encode_columns(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    categorical_cols: List[str],
    drop_first: bool = False,
    train_reference: pd.DataFrame | None = None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Apply one-hot encoding to selected categorical columns.
    If train_reference is provided, new columns will match train columns exactly.
    """
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    ref_columns = {}
    if train_reference is not None:
        ref_columns = pd.get_dummies(train_reference[categorical_cols], drop_first=drop_first).columns.tolist()

    encoded_dfs = []
    for df_idx, df in enumerate(dfs):
        df = df.copy()
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
        if ref_columns:
            for col in ref_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[sorted(df_encoded.columns)]
        encoded_dfs.append(df_encoded)

    print(
        f"One-hot encoding successfully applied to {len(categorical_cols)} columns "
        f"on {len(encoded_dfs)} dataset(s)."
    )
    return encoded_dfs[0] if single_df else encoded_dfs

#--- Function : ordinal_encode_columns ---
def ordinal_encode_columns(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    ordinal_mappings: dict,
    train_reference: pd.DataFrame | None = None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Apply ordinal encoding to selected columns.
    If train_reference is provided, use the same category ordering.
    """
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    ref_mappings = {}
    if train_reference is not None:
        for col, order in ordinal_mappings.items():
            if col not in train_reference.columns:
                raise KeyError(f"Column '{col}' not found in train_reference")
            ref_mappings[col] = {cat: i for i, cat in enumerate(order)}

    encoded_dfs = []
    for df_idx, df in enumerate(dfs):
        df = df.copy()
        for col, order in ordinal_mappings.items():
            mapping = ref_mappings[col] if ref_mappings else {cat: i for i, cat in enumerate(order)}
            invalid_mask = ~df[col].isin(mapping.keys()) & df[col].notna()
            if invalid_mask.any():
                invalid_values = df.loc[invalid_mask, col].unique()
                raise ValueError(f"Invalid values in column '{col}' (dataset {df_idx}): {invalid_values}")
            df[col] = df[col].map(mapping).astype(int)
        encoded_dfs.append(df)

    print(
        f"Ordinal encoding successfully applied to {len(ordinal_mappings)} columns "
        f"on {len(encoded_dfs)} dataset(s)."
    )
    return encoded_dfs[0] if single_df else encoded_dfs
