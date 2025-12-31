import pandas as pd
import numpy as np
from scipy.stats import skew, boxcox, yeojohnson

#---Function : best_transformation---
def best_transformation(series):
    """
    Determine the best transformation to reduce skewness of a numeric series.
    Returns a one-row DataFrame with Column, Best Method, Original Skew, Transformed Skew.
    """
    s = series.dropna()
    transformations = {}

    #Original
    original_skew = skew(s)
    transformations["original"] = (abs(original_skew), s)

    #Log transformation (only for non-negative values)
    if (s >= 0).all():
        log_t = np.log1p(s)
        transformations["log"] = (abs(skew(log_t)), log_t)

    #Square root (only for non-negative values)
    if (s >= 0).all():
        sqrt_t = np.sqrt(s)
        transformations["sqrt"] = (abs(skew(sqrt_t)), sqrt_t)

    #Box-Cox (only for strictly positive values)
    if (s > 0).all():
        bc_t, _ = boxcox(s)
        transformations["boxcox"] = (abs(skew(bc_t)), bc_t)

    #Yeo-Johnson (works for all real values)
    yj_t, _ = yeojohnson(s)
    transformations["yeojohnson"] = (abs(skew(yj_t)), yj_t)

    #Best method
    best_method = min(transformations, key=lambda m: transformations[m][0])
    transformed_skew = transformations[best_method][0]

    #Build DataFrame
    df_res = pd.DataFrame({
        "Column": [series.name if series.name is not None else "Variable"],
        "Best Method": [best_method],
        "Original Skew": [original_skew],
        "Transformed Skew": [transformed_skew]
    })

    return df_res
