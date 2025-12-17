import pandas as pd
import pytest
from preprocessing.missing import missing_stats

#--- Function test_missing_stats_basic_case ---
def test_missing_stats_basic_case():
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": [None, None, 6]
    })

    result = missing_stats(df)

    assert isinstance(result, dict)

    assert result["total_missing"] == 3
    assert result["num_columns_missing"] == 2
    assert result["num_rows_missing"] == 2

    expected_percent = 100 * 3 / (3 * 2)
    assert result["percent_missing"] == expected_percent

#--- Function : test_missing_stats_no_missing ---
def test_missing_stats_no_missing():
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4]
    })

    result = missing_stats(df)

    assert result["total_missing"] == 0
    assert result["percent_missing"] == 0.0
    assert result["num_columns_missing"] == 0
    assert result["num_rows_missing"] == 0
