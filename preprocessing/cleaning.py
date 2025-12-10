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

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing name columns.
    first_col : str, default 'first_name'
        Name of the first name column.
    last_col : str, default 'last_name'
        Name of the last name column.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned 'first_name' and 'last_name' columns.
    """

    #Standardize missing values for both first and last names
    df[first_col] = df[first_col].astype(str).str.strip().replace(['nan', 'None', '', 'NaN', '<na>'], pd.NA)
    df[last_col] = df[last_col].astype(str).str.strip().replace(['nan', 'None', '', 'NaN', '<na>'], pd.NA)

    # --- Helper function: proper_case ---
    def proper_case(name: str) -> str:
        """
        Capitalize names properly, handling:
        - Hyphens (Anne-Marie)
        - Apostrophes (O'Neil)
        - Mc/Mac prefixes (McDonald)
        - Multiple parts in a name
        """
        if pd.isna(name) or str(name).strip() == '':
            return pd.NA

        def cap_part(part: str) -> str:
            #Handle Mc or Mac prefixes (capitalize letter after "Mc")
            part = re.sub(r"\b(Mc)(\w)", lambda m: m.group(1) + m.group(2).upper(), part, flags=re.IGNORECASE)
            
            #Handle apostrophes (capitalize letters around apostrophes)
            part = re.sub(r"(\b\w)'(\w)", lambda m: m.group(1).upper() + "'" + m.group(2).upper(), part)
            
            #Default capitalization for the rest
            return part.capitalize()

        #Split by hyphen, capitalize each sub-part, then join back
        parts = [cap_part(p) for p in name.split('-')]
        return '-'.join(parts)

    #Apply proper capitalization to first and last names
    df[first_col] = df[first_col].apply(proper_case)
    df[last_col] = df[last_col].apply(proper_case)

    #Split multi-part first names if needed
    split_names = df[first_col].str.split(' ', n=1, expand=True)
    df['first_name_clean'] = split_names[0]  # Keep first part as first_name
    df['last_extracted'] = split_names[1] if split_names.shape[1] > 1 else pd.NA  # Extract second part as potential last name

    #Merge with original last name intelligently
    df['last_name_clean'] = df.apply(
        lambda row: row[last_col] 
        if pd.notna(row[last_col]) and str(row[last_col]).strip() != str(row['last_extracted']).strip()
        else row['last_extracted'],
        axis=1
    )

    #Replace residual placeholders or empty strings with pd.NA
    df['last_name_clean'] = df['last_name_clean'].replace(['<na>', '', 'nan', 'None', 'NaN'], pd.NA)

    #Capitalize last name after merging
    df['last_name_clean'] = df['last_name_clean'].apply(proper_case)

    #Drop temporary extracted column
    df.drop(columns=['last_extracted'], inplace=True)

    return df

# --- Function : clean_names_multiple ---
def clean_names_multiple(dfs: Dict[str, pd.DataFrame], first_col: str = 'first_name', last_col: str = 'last_name') -> Dict[str, pd.DataFrame]:
    """
    Apply name cleaning to multiple DataFrames stored in a dictionary.

    Parameters
    ----------
    dfs : dict
        Dictionary of DataFrames {file_name: DataFrame}.
    first_col : str, default 'first_name'
        Name of the first name column.
    last_col : str, default 'last_name'
        Name of the last name column.

    Returns
    -------
    dict
        Dictionary with cleaned DataFrames.
    """
    for key, df in dfs.items():
        df = clean_names(df, first_col=first_col, last_col=last_col)
        # Replace original columns with cleaned ones
        df.drop(columns=[first_col, last_col], inplace=True)
        df.rename(columns={'first_name_clean': 'first_name', 'last_name_clean': 'last_name'}, inplace=True)
        dfs[key] = df

    return dfs
