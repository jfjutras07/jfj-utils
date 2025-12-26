import pandas as pd
import numpy as np
from scipy import stats
from eda.stats_non_param import kruskal_wallis_test

#--- Function : test_kruskal_wallis_basic ---
def test_kruskal_wallis_basic():
    df = pd.DataFrame({
        "score": [10,12,14,20,22,24,30,32,34],
        "group": ["A"]*3 + ["B"]*3 + ["C"]*3
    })
    stat, p_value = kruskal_wallis_test(df, "score", "group")
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

#--- Function : test_kruskal_wallis_identical_groups ---
def test_kruskal_wallis_identical_groups():
    df = pd.DataFrame({
        "score": [1,1,1,2,2,2,3,3,3],
        "group": ["A"]*3 + ["B"]*3 + ["C"]*3
    })
    stat, p_value = kruskal_wallis_test(df, "score", "group")
    # H statistic should be non-negative
    assert stat >= 0

#--- Function : test_kruskal_wallis_non_numeric_raises ---
def test_kruskal_wallis_non_numeric_raises():
    df = pd.DataFrame({
        "score": ["a","b","c","d","e","f","g","h","i"],
        "group": ["A"]*3 + ["B"]*3 + ["C"]*3
    })
    try:
        kruskal_wallis_test(df, "score", "group")
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True
