import pandas as pd
import numpy as np
from eda.multicollinearity import check_multicollinearity
import pytest

#--- Function : test_check_multicollinearity_basic ---
def test_check_multicollinearity_basic():
    df = pd.DataFrame({
        "x1": np.random.rand(10),
        "x2": np.random.rand(10),
        "x3": np.random.rand(10)
    })
    results = check_multicollinearity(df)

    assert isinstance(results, dict)
    assert "correlation_matrix" in results
    assert "vif" in results
    assert isinstance(results["correlation_matrix"], pd.DataFrame)
    assert isinstance(results["vif"], pd.DataFrame)
    assert results["correlation_matrix"].shape[0] == df.shape[1]
    assert results["vif"].shape[0] == df.shape[1]

#--- Function : test_check_multicollinearity_method_parameter ---
def test_check_multicollinearity_method_parameter():
    df = pd.DataFrame({
        "a": np.random.rand(5),
        "b": np.random.rand(5)
    })
    results_pearson = check_multicollinearity(df, method='pearson')
    results_spearman = check_multicollinearity(df, method='spearman')

    assert results_pearson["correlation_matrix"].shape == results_spearman["correlation_matrix"].shape

#--- Function : test_check_multicollinearity_non_numeric_error ---
def test_check_multicollinearity_non_numeric_error():
    df = pd.DataFrame({
        "num": np.random.rand(5),
        "cat": ["a", "b", "c", "d", "e"]
    })
    try:
        check_multicollinearity(df)
        assert False  # Should not reach here
    except ValueError as e:
        assert str(e) == "Input DataFrame must contain only numerical variables."

#--- Function : test_check_multicollinearity_invalid_method_error ---
def test_check_multicollinearity_invalid_method_error():
    df = pd.DataFrame({
        "x1": np.random.rand(5),
        "x2": np.random.rand(5)
    })
    try:
        check_multicollinearity(df, method='invalid')
        assert False
    except ValueError as e:
        assert str(e) == "Method must be 'spearman' or 'pearson'."
