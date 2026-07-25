import re
import pandas as pd
import numpy as np

#--- Function : date_quality ---
def date_quality(df, date_cols):
    """
    Diagnoses date columns by detecting format ambiguities,
    invalid values, mixed date conventions, and conversion issues.

    Parameters:
    - df: pandas DataFrame
    - date_cols: list of column names containing date strings

    Returns:
    - diagnostics: dictionary containing quality indicators for each date column
    """

    diagnostics = {}

    for col in date_cols:

        if col not in df.columns:
            continue

        values = df[col].dropna().astype(str)

        day_first = 0
        month_first = 0
        ambiguous = 0
        invalid = 0

        for value in values:

            parts = re.split(r"[/-]", value)

            if len(parts) != 3:
                invalid += 1
                continue

            try:
                first = int(parts[0])
                second = int(parts[1])

                if first > 12:
                    day_first += 1

                elif second > 12:
                    month_first += 1

                else:
                    ambiguous += 1

            except ValueError:
                invalid += 1

        total = len(values)

        diagnostics[col] = {
            "total_values": total,
            "day_first_dates": day_first,
            "month_first_dates": month_first,
            "ambiguous_dates": ambiguous,
            "invalid_dates": invalid,
            "mixed_format_detected": day_first > 0 and month_first > 0,
            "missing_values": df[col].isna().sum(),
            "missing_percentage": round(df[col].isna().mean() * 100, 2)
        }

    return diagnostics

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
