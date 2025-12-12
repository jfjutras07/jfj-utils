import os
from typing import Dict, Optional
import pandas as pd

# --- Function: read_table ---
def read_table(file_path: str, encoding: str = 'utf-8', on_bad_lines: str = 'skip') -> Optional[pd.DataFrame]:
    """
    Read a single CSV, Excel, Parquet, or JSON file into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return None

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, sep=None, engine='python', encoding=encoding, on_bad_lines=on_bad_lines)
            if df.shape[1] == 1:
                sep = ';' if ';' in df.iloc[0,0] else (',' if ',' in df.iloc[0,0] else '\t')
                df = df.iloc[:,0].str.split(sep, expand=True)
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        elif ext == ".json":
            df = pd.read_json(file_path)
        else:
            print(f"Unsupported file type: {ext}")
            return None
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


# --- Function: read_folder ---
def read_folder(folder_path: str, file_types: list = None, encoding: str = 'utf-8', on_bad_lines: str = 'skip') -> Dict[str, pd.DataFrame]:
    """
    Load all tabular files from a folder into a dictionary of pandas DataFrames.
    """
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return {}

    all_files = os.listdir(folder_path)
    dfs = {}

    for file in all_files:
        ext = os.path.splitext(file)[1].lower()
        if file_types and ext not in file_types:
            continue
        if ext not in [".csv", ".xls", ".xlsx", ".parquet", ".json"]:
            continue

        path = os.path.join(folder_path, file)
        name = os.path.splitext(file)[0]
        df = read_table(path, encoding=encoding, on_bad_lines=on_bad_lines)
        if df is not None:
            dfs[name] = df
            print(f"Loaded {name}, shape: {df.shape}")
        else:
            print(f"Failed to load {name}")

    return dfs


# --- Function: check_data ---
def check_data(data, n=5):
    """
    Perform a preliminary exploration of a pandas DataFrame or a dict of DataFrames.
    """
    if isinstance(data, dict):
        for name, df in data.items():
            print(f"\n=== {name} ===")
            _check_single_df(df, n)
    else:
        _check_single_df(data, n)

def _check_single_df(df, n=5):
    """Helper function for checking a single DataFrame."""
    print("\nColumns:", df.columns.to_list(), "\n")
    print("Shape:", df.shape, "\n")
    print("Data types:\n", df.dtypes, "\n")
    print("Missing values:\n", df.isnull().sum(), "\n")
    print("Duplicates:", df.duplicated().sum(), "\n")
    print("\nInfo:")
    df.info()
