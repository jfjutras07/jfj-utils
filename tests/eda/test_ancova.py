import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm
from eda.stats_param import ancova_test

#--- Function : test_ancova_test_basic ---
def test_ancova_test_basic():
    df = pd.DataFrame({
        'score': [85, 90, 88, 92, 78, 80, 82, 75],
        'school': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'hours_study': [5, 6, 5, 7, 4, 5, 4, 3]
    })
    anova_table = ancova_test(df, dv='score', factor='school', covariates=['hours_study'])
    assert isinstance(anova_table, pd.DataFrame)
    assert "F" in anova_table.columns
    assert "PR(>F)" in anova_table.columns

#--- Function : test_ancova_test_non_numeric_dv_raises ---
def test_ancova_test_non_numeric_dv_raises():
    df = pd.DataFrame({
        'score': ['a','b','c','d'],
        'school': ['A','A','B','B'],
        'hours_study': [5,6,4,3]
    })
    try:
        ancova_test(df, dv='score', factor='school', covariates=['hours_study'])
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True

#--- Function : test_ancova_test_non_numeric_covariate_raises ---
def test_ancova_test_non_numeric_covariate_raises():
    df = pd.DataFrame({
        'score': [85,90,88,92],
        'school': ['A','A','B','B'],
        'hours_study': ['a','b','c','d']
    })
    try:
        ancova_test(df, dv='score', factor='school', covariates=['hours_study'])
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True
