import pandas as pd
import re
from typing import Dict

#--- Function: clean_names ---
def clean_names(df: pd.DataFrame, first_col: str='first_name', last_col: str='last_name') -> pd.DataFrame:
    """
    Clean first and last name columns in a DataFrame.
    Steps:
    - Strip whitespace and standardize missing values
    - Proper capitalization (hyphens, apostrophes, Mc/Mac prefixes)
    - Split multi-part first names if needed
    - Merge extracted last name with original last name
    """
    #Standardize missing values
    df[first_col] = df[first_col].astype(str).str.strip().replace(['nan','None','', 'NaN','<na>'], pd.NA)
    df[last_col] = df[last_col].astype(str).str.strip().replace(['nan','None','', 'NaN','<na>'], pd.NA)

    #--- Helper: proper_case ---
    def proper_case(name: str) -> str:
        if pd.isna(name) or str(name).strip() == '':
            return pd.NA
        def cap_part(part: str) -> str:
            part = re.sub(r"\b(Mc)(\w)", lambda m: m.group(1)+m.group(2).upper(), part, flags=re.IGNORECASE)
            part = re.sub(r"(\b\w)'(\w)", lambda m: m.group(1).upper()+"'"+m.group(2).upper(), part)
            return part.capitalize()
        parts = [cap_part(p) for p in name.split('-')]
        return '-'.join(parts)

    #Apply capitalization
    df[first_col] = df[first_col].apply(proper_case)
    df[last_col] = df[last_col].apply(proper_case)

    #Split first names
    split_names = df[first_col].str.split(' ', n=1, expand=True)
    df['first_name_clean'] = split_names[0]
    df['last_extracted'] = split_names[1] if split_names.shape[1] > 1 else pd.NA

    #Merge last name
    df['last_name_clean'] = df.apply(
        lambda row: row[last_col] if pd.notna(row[last_col]) and str(row[last_col]).strip()!=str(row['last_extracted']).strip() else row['last_extracted'],
        axis=1
    )

    #Ensure missing values are pd.NA
    df['last_name_clean'] = df['last_name_clean'].replace([None,'','nan','None','<na>'], pd.NA)

    #Capitalize last name
    df['last_name_clean'] = df['last_name_clean'].apply(proper_case)

    df.drop(columns=['last_extracted'], inplace=True)
    return df

#--- Function: clean_names_multiple ---
def clean_names_multiple(dfs: Dict[str,pd.DataFrame], first_col: str='first_name', last_col: str='last_name') -> Dict[str,pd.DataFrame]:
    """
    Apply clean_names to multiple DataFrames in a dictionary.
    """
    for key, df in dfs.items():
        df_clean = clean_names(df, first_col=first_col, last_col=last_col)

        #Drop original columns
        df_clean.drop(columns=[first_col,last_col], inplace=True)

        #Rename cleaned columns
        df_clean.rename(columns={'first_name_clean':'first_name','last_name_clean':'last_name'}, inplace=True)

        #Final fix: all missing last_name values must be pd.NA
        df_clean['last_name'] = df_clean['last_name'].apply(lambda x: pd.NA if pd.isna(x) or str(x).strip()=='' else x)

        dfs[key] = df_clean

    return dfs
