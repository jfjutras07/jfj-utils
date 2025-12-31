import pandas as pd
import numpy as np
from scipy.stats import skew, boxcox, yeojohnson

def best_transformation_df(series, col_name):
    """
    Return a DataFrame with best transformation info for a single series.
    """
    s = series.dropna()
    transformations = {}

    # Original
    original_skew = skew(s)
    transformations["original"] = (abs(original_skew), s)

    # Log transformation
    if (s >= 0).all():
        log_t = np.log1p(s)
        transformations["log"] = (abs(skew(log_t)), log_t)

    # Square root
    if (s >= 0).all():
        sqrt_t = np.sqrt(s)
        transformations["sqrt"] = (abs(skew(sqrt_t)), sqrt_t)

    # Box-Cox
    if (s > 0).all():
        bc_t, _ = boxcox(s)
        transformations["boxcox"] = (abs(skew(bc_t)), bc_t)

    # Yeo-Johnson
    yj_t, _ = yeojohnson(s)
    transformations["yeojohnson"] = (abs(skew(yj_t)), yj_t)

    # Best method
    best_method = min(transformations, key=lambda m: transformations[m][0])
    transformed_skew = transformations[best_method][0]

    # Build DataFrame
    df_res = pd.DataFrame({
        "Column": [col_name],
        "Best Method": [best_method],
        "Original Skew": [original_skew],
        "Transformed Skew": [transformed_skew]
    })

    return df_res

# --- Function : best_transformation_for_df ---
def best_transformation_for_df(df, numeric_cols):
    """
    Apply best_transformation to multiple numeric columns in a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - numeric_cols: list of numeric column names to analyze

    Returns:
    - results_df: DataFrame with columns: Column, Best Method, Original Skew, Transformed Skew
    """
    results = []
    for col in numeric_cols:
        best_method, orig_skew, trans_skew = best_transformation(df[col])
        results.append({
            "Column": col,
            "Best Method": best_method,
            "Original Skew": orig_skew,
            "Transformed Skew": trans_skew
        })

    results_df = pd.DataFrame(results)
    return results_df
