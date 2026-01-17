import pandas as pd
from typing import Dict, List, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin

#--- Class : categorical_encoder ---
class categorical_encoder(BaseEstimator, TransformerMixin):
    """
    All-in-one encoder for Binary, Ordinal, and One-Hot encoding.
    Automatically casts encoded columns to int where relevant.
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

        # Apply Manual Mappings (Binary & Ordinal)
        for col, mapping in self.mapping_rules.items():
            if col in X.columns:
                initial_na = X[col].isna()
                X[col] = X[col].map(mapping)

                if self.strict_mapping:
                    invalid_mask = X[col].isna() & ~initial_na
                    if invalid_mask.any():
                        invalid_vals = X.loc[invalid_mask, col].unique()
                        raise ValueError(f"Mapping error in '{col}': {invalid_vals} not found in rules.")

                # Type cast to int after mapping
                X[col] = X[col].astype(int)

        # Apply One-Hot Encoding
        if self.one_hot_cols:
            X_oh = pd.get_dummies(X, columns=self.one_hot_cols, drop_first=self.drop_first)

            # Identify columns not part of the new One-Hot features
            other_cols = [c for c in X_oh.columns if c not in self.one_hot_features_]

            # Reindex One-Hot part and combine
            X_oh_part = X_oh.reindex(columns=list(other_cols) + list(self.one_hot_features_), fill_value=0)

            # Type cast boolean columns to int
            bool_cols = X_oh_part.select_dtypes(include='bool').columns
            X_oh_part[bool_cols] = X_oh_part[bool_cols].astype(int)

            return X_oh_part

        return X

#--- Function : binary_encode_columns ---
def binary_encode_columns(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    binary_mappings: Dict[str, Dict],
    strict: bool = True,
    train_reference: pd.DataFrame | None = None
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

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

            initial_na = df[col].isna()
            df[col] = df[col].map(mapping)

            invalid_mask = df[col].isna() & ~initial_na
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

            # Type cast to int
            df[col] = df[col].astype(int)

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
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

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

            if ref_mappings:
                df[col] = df[col].map(ref_mappings[col]).fillna(-1).astype(int)
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
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    final_column_structure = None
    if train_reference is not None:
        ref_dummy = pd.get_dummies(train_reference, columns=categorical_cols, drop_first=drop_first)
        final_column_structure = ref_dummy.columns

    encoded_dfs = []
    for df_idx, df in enumerate(dfs):
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)

        if final_column_structure is not None:
            df_encoded = df_encoded.reindex(columns=final_column_structure, fill_value=0)

        # Type cast boolean columns to int
        bool_cols = df_encoded.select_dtypes(include='bool').columns
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

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
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    ref_mappings = {}
    for col, order in ordinal_mappings.items():
        ref_mappings[col] = {cat: i for i, cat in enumerate(order)}

    encoded_dfs = []
    for df_idx, df in enumerate(dfs):
        df = df.copy()
        for col, mapping in ref_mappings.items():
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in dataset {df_idx}")

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
