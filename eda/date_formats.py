import re
import pandas as pd
import numpy as np

#--- Function : detect_date_patterns ---
def detect_date_patterns(df, date_cols):
    """
    Detects the structural patterns of date strings in a DataFrame.
    
    Parameters:
    - df: pandas DataFrame
    - date_cols: list of column names containing date strings
    
    Returns:
    - patterns_dict: dict where keys are column names and values are sets of detected patterns
    """
    def extract_pattern(date_str):
        pattern = ""
        for c in str(date_str):
            if c.isdigit():
                pattern += "d"
            elif c.isalpha():
                pattern += "a"
            else:
                pattern += c
        return pattern

    patterns_dict = {}

    for col in date_cols:
        if col not in df.columns:
            continue
        
        unique_dates = df[col].dropna().astype(str).unique()
        patterns_found = set(extract_pattern(val) for val in unique_dates)
        patterns_dict[col] = patterns_found
    
    return patterns_dict
