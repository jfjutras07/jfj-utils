import pandas as pd
import numpy as np
from scipy import stats
from eda.stats_param import paired_ttest

#--- Function : test_paired_ttest_basic ---
def test_paired_ttest_basic():
    df = pd.DataFrame({
        "before": [10, 12, 14, 16, 18],
        "after":  [11, 13, 15, 17, 19]
    })
    t_stat, p_value = paired_ttest(df, "before", "after")
    assert isinstance(t_stat, float)
    assert isinstance(p_value, float)

#--- Function : test_paired_ttest_identical_columns ---
def test_paired_ttest_identical_columns():
    df = pd.DataFrame({
        "before": [1, 2, 3, 4],
        "after":  [1, 2, 3, 4]
    })
    t_stat, p_value = paired_ttest(df, "before", "after")
    # Identical columns => t ≈ 0, p ≈ 1
    assert np.isclose(t_stat, 0.0)
    assert np.isclose(p_value, 1.0)

#--- Function : test_paired_ttest_non_numeric_raises ---
def test_paired_ttest_non_numeric_raises():
    df = pd.DataFrame({
        "before": ["a", "b", "c"],
        "after": ["x", "y", "z"]
    })
    try:
        paired_ttest(df, "before", "after")
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True

#--- Function : test_paired_ttest_mismatched_length_raises ---
def test_paired_ttest_mismatched_length_raises():
    df = pd.DataFrame({
        "before": [1, 2, 3],
        "after": [4, 5]
    })
    try:
        paired_ttest(df, "before", "after")
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True
