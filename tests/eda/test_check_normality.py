import pandas as pd
import numpy as np
from eda.check_normality import normality_check

#--- Function : test_normality_check_basic ---
def test_normality_check_basic():
    df = pd.DataFrame({
        "a": np.random.normal(0, 1, 50),
        "b": np.random.uniform(0, 1, 50),
        "c": np.arange(50)
    })
    results = normality_check(df)
    
    assert isinstance(results, pd.DataFrame)
    assert all(col in results.columns for col in [
        "Column", "N", "Mean", "Std", "Shapiro-Wilk",
        "D’Agostino K²", "Anderson-Darling (5%)", "Kolmogorov-Smirnov"
    ])
    assert results.shape[0] == 3

#--- Function : test_normality_check_numeric_cols ---
def test_normality_check_numeric_cols():
    df = pd.DataFrame({
        "a": np.random.normal(0, 1, 30),
        "b": np.random.normal(10, 2, 30),
        "c": ["x"]*30
    })
    results = normality_check(df, numeric_cols=["a", "b"])
    
    assert results.shape[0] == 2
    assert set(results["Column"]) == {"a", "b"}

#--- Function : test_normality_check_flags_types ---
def test_normality_check_flags_types():
    df = pd.DataFrame({
        "a": np.random.normal(0, 1, 25)
    })
    results = normality_check(df)
    
    flags = ["Shapiro-Wilk", "D’Agostino K²", "Anderson-Darling (5%)", "Kolmogorov-Smirnov"]
    for flag in flags:
        val = results[flag].iloc[0]
        assert val in [True, False, np.nan]

#--- Function : test_normality_check_empty_dataframe ---
def test_normality_check_empty_dataframe():
    df = pd.DataFrame()
    results = normality_check(df)
    
    assert results.empty
