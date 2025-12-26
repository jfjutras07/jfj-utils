import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm
from eda.stats_param import anova_test

#--- Function : test_anova_test_basic ---
def test_anova_test_basic():
    df = pd.DataFrame({
        "score": [10, 12, 14, 20, 22, 24, 30, 32, 34],
        "group": ["A"]*3 + ["B"]*3 + ["C"]*3
    })
    anova_table = anova_test(df, "score", "group")
    assert isinstance(anova_table, pd.DataFrame)
    assert "F" in anova_table.columns
    assert "PR(>F)" in anova_table.columns

#--- Function : test_anova_test_identical_groups ---
def test_anova_test_identical_groups():
    df = pd.DataFrame({
        "score": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "group": ["A"]*3 + ["B"]*3 + ["C"]*3
    })
    anova_table = anova_test(df, "score", "group")
    f_stat = anova_table["F"].iloc[0]
    # F should be non-negative
    assert f_stat >= 0

#--- Function : test_anova_test_non_numeric_raises ---
def test_anova_test_non_numeric_raises():
    df = pd.DataFrame({
        "score": ["a", "b", "c", "d", "e", "f"],
        "group": ["A", "A", "B", "B", "C", "C"]
    })
    try:
        anova_test(df, "score", "group")
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True
