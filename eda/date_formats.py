import re
import pandas as pd
import numpy as np

# --- Function: detect_date_patterns ---
def detect_date_patterns(df, date_cols):
    """
    Detect structural date patterns and identify potential
    date format ambiguities (MM/DD/YYYY vs DD/MM/YYYY).

    Parameters
    ----------
    df : pandas.DataFrame
    date_cols : list
        List of columns containing date strings.

    Returns
    -------
    dict
        Dictionary containing structural patterns and
        detected date format information for each column.
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

    results = {}

    for col in date_cols:

        if col not in df.columns:
            continue

        values = df[col].dropna().astype(str)

        patterns = sorted({extract_pattern(v) for v in values})

        day_first = 0
        month_first = 0
        ambiguous = 0

        for v in values:

            parts = re.split(r"[/-]", v)

            if len(parts) != 3:
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
                continue

        results[col] = {
            "patterns": patterns,
            "day_first_detected": day_first,
            "month_first_detected": month_first,
            "ambiguous_dates": ambiguous,
            "mixed_format": day_first > 0 and month_first > 0
        }

    return results
