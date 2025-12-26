import pandas as pd
import numpy as np
from scipy import stats
from eda.stats_param import two_sample_ttest


#--- Function : test_two_sample_ttest_basic ---
def test_two_sample_ttest_basic():
    df = pd.DataFrame({
        "value": [1, 2, 3, 4, 5, 6, 7, 8],
        "group": ["A"]*4 + ["B"]*4
    })
    t_stat, p_value = two_sample_ttest(df, "value", "group", "A", "B")
    assert isinstance(t_stat, float)
    assert isinstance(p_value, float)


#--- Function : test_two_sample_ttest_identical_groups ---
def test_two_sample_ttest_identical_groups():
    df = pd.DataFrame({
        "value": [1, 1, 1, 1, 1, 1, 1, 1],
        "group": ["A"]*4 + ["B"]*4
    })
    t_stat, p_value = two_sample_ttest(df, "value", "group", "A", "B")
    # Identical groups => t ≈ 0, p ≈ 1
    assert np.isclose(t_stat, 0.0)
    assert np.isclose(p_value, 1.0)


#--- Function : test_two_sample_ttest_non_numeric_raises ---
def test_two_sample_ttest_non_numeric_raises():
    df = pd.DataFrame({
        "value": ["a","b","c","d","e","f","g","h"],
        "group": ["A"]*4 + ["B"]*4
    })
    try:
        two_sample_ttest(df, "value", "group", "A", "B")
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True
