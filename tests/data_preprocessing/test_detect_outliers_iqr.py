import pandas as pd
import numpy as np
from data_preprocessing.outliers import detect_outliers_iqr

#--- Function : test_detect_outliers_iqr_basic ---
def test_detect_outliers_iqr_basic():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 100],
        "b": [10, 12, 11, 13, 12]
    })

    results = detect_outliers_iqr(df, columns=["a", "b"])

    assert isinstance(results, dict)
    assert "a" in results
    assert "b" in results
    assert "Total_outliers" in results
    assert results["a"] > 0
    assert results["b"] == 0
    assert results["Total_outliers"] == results["a"] + results["b"]

#--- Function : test_detect_outliers_iqr_custom_factor ---
def test_detect_outliers_iqr_custom_factor():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 10]
    })

    results_default = detect_outliers_iqr(df, columns=["x"], factor=1.5)
    results_strict = detect_outliers_iqr(df, columns=["x"], factor=0.5)

    assert results_strict["x"] >= results_default["x"]

#--- Function : test_detect_outliers_iqr_missing_column ---
def test_detect_outliers_iqr_missing_column():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5]
    })

    results = detect_outliers_iqr(df, columns=["a", "missing"])

    assert "a" in results
    assert "missing" not in results
    assert results["Total_outliers"] == results["a"]
