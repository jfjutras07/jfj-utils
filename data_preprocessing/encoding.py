import pandas as pd
from typing import Dict, List, Union

# --- Function : binary_encode_columns ---
def binary_encode_columns(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    binary_mappings: Dict[str, Dict],
    strict: bool = True
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Apply explicit binary encoding to selected columns for one or multiple datasets
    and perform consistency checks.

    Parameters:
        dfs : pd.DataFrame or list of pd.DataFrame
            Dataset(s) on which binary encoding will be applied
        binary_mappings : dict
            Dictionary of the form:
            {
                'column_name': {'positive_class': 1, 'negative_class': 0}
            }
        strict : bool, default=True
            If True, raises an error when unmapped or invalid values are found.
            If False, leaves NaNs and prints warnings.

    Returns:
        pd.DataFrame or list of pd.DataFrame
            Binary-encoded dataset(s)
    """
    #Ensure dfs is iterable
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    encoded_dfs = []

    for df_idx, df in enumerate(dfs):
        df = df.copy()

        for col, mapping in binary_mappings.items():

            #Check column existence
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in dataset {df_idx}")

            #Apply mapping
            df[col] = df[col].map(mapping)

            #Sanity check: only {0,1}
            invalid_mask = ~df[col].isin([0, 1]) & df[col].notna()

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

    # --- Validation message ---
    print(
        f"Binary encoding successfully applied to "
        f"{len(binary_mappings)} columns on {len(encoded_dfs)} dataset(s)."
    )

    return encoded_dfs[0] if single_df else encoded_dfs

import pandas as pd
from typing import List, Union

# --- Function : one_hot_encode_columns ---
def one_hot_encode_columns(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    categorical_cols: List[str],
    drop_first: bool = False
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Apply one-hot encoding to selected categorical columns for one or multiple datasets.

    Parameters:
        dfs : pd.DataFrame or list of pd.DataFrame
            Dataset(s) on which one-hot encoding will be applied
        categorical_cols : list of str
            List of columns to one-hot encode
        drop_first : bool, default=False
            If True, drops the first category to avoid multicollinearity

    Returns:
        pd.DataFrame or list of pd.DataFrame
            One-hot encoded dataset(s)
    """
    #Ensure dfs is iterable
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    encoded_dfs = []

    for df_idx, df in enumerate(dfs):
        df = df.copy()

        #Check column existence
        missing_cols = [col for col in categorical_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(
                f"Columns not found in dataset {df_idx}: {missing_cols}"
            )

        #Apply one-hot encoding
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)

        encoded_dfs.append(df)

    # --- Validation message ---
    print(
        f"One-hot encoding successfully applied to {len(categorical_cols)} columns "
        f"on {len(encoded_dfs)} dataset(s)."
    )

    return encoded_dfs[0] if single_df else encoded_dfs
