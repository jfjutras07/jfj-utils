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
    # Ensure dfs is iterable
    single_df = False
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        single_df = True

    encoded_dfs = []

    for df_idx, df in enumerate(dfs):
        df = df.copy()

        for col, mapping in binary_mappings.items():

            # Check column existence
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in dataset {df_idx}")

            # Apply mapping
            df[col] = df[col].map(mapping)

            # Sanity check: only {0,1}
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
