import pandas as pd
from typing import Dict
import re

# --- Function : clean_names ---
def clean_names(df: pd.DataFrame, first_col: str = 'first_name', last_col: str = 'last_name') -> pd.DataFrame:
    """
    Clean first and last name columns in a DataFrame.

    Cleaning steps:
    - Strip whitespace and standardize missing values
    - Proper capitalization (handles hyphens, apostrophes, Mc/Mac prefixes)
    - Split multi-part first names if needed
    - Merge extracted last name with original last name intelligently
    """
    #Standardize missing values
    df[first_col] = df[first_col].astype(str).str.strip().replace(['nan', 'None', '', 'NaN', '<na>'], pd.NA)
    df[last_col] = df[last_col].astype(str).str.strip().replace(['nan', 'None', '', 'NaN', '<na>'], pd.NA)

    #Helper function: proper_case
    def proper_case(name: str) -> str:
        if pd.isna(name) or str(name).strip() == '':
            return pd.NA

        def cap_part(part: str) -> str:
            # Handle Mc/Mac prefixes
            part = re.sub(r'\b(Mc)(\w)', lambda m: m.group(1) + m.group(2).upper(), part, flags=re.IGNORECASE)
            # Handle apostrophes
            part = re.sub(r"(\b\w)'(\w)", lambda m: m.group(1).upper() + "'" + m.group(2).upper(), part)
            # Capitalize each sub-part (default)
            return '-'.join([p[0].upper() + p[1:].lower() if len(p) > 1 else p.upper() for p in part.split('-')])

        #Split by hyphen, capitalize each sub-part, join back
        return '-'.join([cap_part(p) for p in name.split('-')])

    #Apply capitalization
    df[first_col] = df[first_col].apply(proper_case)
    df[last_col] = df[last_col].apply(proper_case)

    #Split multi-part first names
    split_names = df[first_col].str.split(' ', n=1, expand=True)
    df['first_name_clean'] = split_names[0]
    df['last_extracted'] = split_names[1] if split_names.shape[1] > 1 else pd.NA

    #Merge intelligently with last name
    df['last_name_clean'] = df.apply(
        lambda row: row[last_col] if pd.notna(row[last_col]) and str(row[last_col]).strip() != str(row['last_extracted']).strip()
        else row['last_extracted'],
        axis=1
    )

    #Replace residual placeholders with pd.NA
    df['last_name_clean'] = df['last_name_clean'].replace(['<na>', '', 'nan', 'None', 'NaN'], pd.NA)
    df['last_name_clean'] = df['last_name_clean'].apply(proper_case)

    #Drop temporary column
    df.drop(columns=['last_extracted'], inplace=True)

    return df

# --- Function : clean_names_multiple ---
def clean_names_multiple(dfs: Dict[str, pd.DataFrame], first_col: str = 'first_name', last_col: str = 'last_name') -> Dict[str, pd.DataFrame]:
    """
    Apply name cleaning to multiple DataFrames stored in a dictionary.
    """
    for key, df in dfs.items():
        df_clean = clean_names(df, first_col=first_col, last_col=last_col)
        #Replace original columns
        df_clean.drop(columns=[first_col, last_col], inplace=True)
        df_clean.rename(columns={'first_name_clean': 'first_name', 'last_name_clean': 'last_name'}, inplace=True)
        dfs[key] = df_clean

    return dfs
