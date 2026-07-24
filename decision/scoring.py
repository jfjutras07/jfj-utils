import pandas as pd
import numpy as np

#--- Function : weighted_score ---
def weighted_score(df, criteria_weights):
    """
    Calculate weighted scores for decision alternatives.

    Parameters
    ----------
    df : pandas.DataFrame
        Decision matrix where rows are alternatives and columns are criteria.

    criteria_weights : dict
        Dictionary containing criteria names and their weights.

    Returns
    -------
    DataFrame
        Alternatives with calculated weighted scores.
    """

    weights = pd.Series(criteria_weights)

    missing_columns = set(weights.index) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing criteria columns: {missing_columns}")

    weighted_values = df[weights.index] * weights

    df_result = df.copy()
    df_result["Weighted_Score"] = weighted_values.sum(axis=1)

    df_result = df_result.sort_values(
        by="Weighted_Score",
        ascending=False
    )

    print("--- Weighted Score Summary ---")
    print(f"Number of alternatives : {len(df_result)}")
    print(f"Criteria used          : {len(criteria_weights)}")
    print("\nRanking:")
    print(df_result[["Weighted_Score"]].to_string())
    print("-" * 35)

    return df_result
