import pandas as pd
import numpy as np
from scipy import stats
from eda.stats_non_param import paired_wilcoxon

#--- Function : test_paired_wilcoxon_basic ---
def test_paired_wilcoxon_basic():
    df = pd.DataFrame({
        "before": [10, 12, 14, 16],
        "after":  [11, 13, 15, 17]
    })
    stat, p_value = paired_wilcoxon(df, "before", "after")
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

#--- Function : test_paired_wilcoxon_identical_columns ---
def test_paired_wilcoxon_identical_columns():
    df = pd.DataFrame({
        "before": [1, 2, 3],
        "after":  [1, 2, 3]
    })
    stat, p_value = paired_wilcoxon(df, "before", "after")
    # Identical columns => statistic ≈ 0, p ≈ 1
    assert np.isclose(stat, 0.0)
    assert np.isclose(p_value, 1.0)

#--- Function : test_paired_wilcoxon_non_numeric_raises ---
def test_paired_wilcoxon_non_numeric_raises():
    df = pd.DataFrame({
        "before": ["a", "b", "c"],
        "after": ["x", "y", "z"]
    })
    try:
        paired_wilcoxon(df, "before", "after")
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True

#--- Function : test_paired_wilcoxon_mismatched_length_raises ---
def test_paired_wilcoxon_mismatched_length_raises():
    df = pd.DataFrame({
        "before": [1, 2, 3],
        "after": [4, 5]
    })
    try:
        paired_wilcoxon(df, "before", "after")
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True
