import pandas as pd
import numpy as np
from eda.best_transformation import best_transformation, best_transformation_for_df

#--- Function : test_best_transformation_basic ---
def test_best_transformation_basic():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    method, orig_skew, trans_skew = best_transformation(s)
    
    assert isinstance(method, str)
    assert isinstance(orig_skew, float)
    assert isinstance(trans_skew, float)
    assert trans_skew >= 0

#--- Function : test_best_transformation_handles_negative ---
def test_best_transformation_handles_negative():
    s = pd.Series([-5, -3, -1, 0, 1, 3, 5])
    method, orig_skew, trans_skew = best_transformation(s)
    
    # Box-Cox should not be chosen (requires strictly positive)
    assert method != "boxcox"
    assert trans_skew >= 0

#--- Function : test_best_transformation_for_df_basic ---
def test_best_transformation_for_df_basic():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50],
        "c": [-5, -3, 0, 3, 5]
    })
    numeric_cols = ["a", "b", "c"]
    results = best_transformation_for_df(df, numeric_cols)
    
    assert isinstance(results, pd.DataFrame)
    assert all(col in results.columns for col in ["Column", "Best Method", "Original Skew", "Transformed Skew"])
    assert results.shape[0] == len(numeric_cols)

#--- Function : test_best_transformation_for_df_skew_values ---
def test_best_transformation_for_df_skew_values():
    df = pd.DataFrame({
        "x": [1, 2, 2, 3, 10],
        "y": [5, 5, 5, 5, 5]
    })
    numeric_cols = ["x", "y"]
    results = best_transformation_for_df(df, numeric_cols)
    
    # Check transformed skew is less than or equal to original
    for _, row in results.iterrows():
        assert row["Transformed Skew"] <= abs(row["Original Skew"])
