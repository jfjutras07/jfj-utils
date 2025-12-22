import pandas as pd
from ingestion.readers import winsorize_columns

#--- Function : test_winsorize_columns_basic ---
def test_winsorize_columns_basic():
    df = pd.DataFrame({
        "a": [1, 2, 3, 100],
        "b": [10, 20, 30, 40]
    })

    result = winsorize_columns(df, columns=["a"], limits=(0.25, 0.25))

    assert isinstance(result, pd.DataFrame)
    assert result is not df
    assert result["a"].max() < 100
    assert result["b"].equals(df["b"])

#--- Function : test_winsorize_columns_multiple_columns ---
def test_winsorize_columns_multiple_columns():
    df = pd.DataFrame({
        "a": [1, 2, 3, 100],
        "b": [5, 6, 7, 200]
    })

    result = winsorize_columns(df, columns=["a", "b"], limits=(0.25, 0.25))

    assert result["a"].max() < 100
    assert result["b"].max() < 200

#--- Function : test_winsorize_columns_column_not_found ---
def test_winsorize_columns_column_not_found():
    df = pd.DataFrame({
        "a": [1, 2, 3]
    })

    result = winsorize_columns(df, columns=["b"])

    assert result.equals(df)

#--- Function : test_winsorize_columns_non_numeric ---
def test_winsorize_columns_non_numeric():
    df = pd.DataFrame({
        "a": ["x", "y", "z"]
    })

    result = winsorize_columns(df, columns=["a"])

    assert result.equals(df)

#--- Function : test_winsorize_columns_original_unchanged ---
def test_winsorize_columns_original_unchanged():
    df = pd.DataFrame({
        "a": [1, 2, 3, 100]
    })

    _ = winsorize_columns(df, columns=["a"])

    assert df["a"].max() == 100
