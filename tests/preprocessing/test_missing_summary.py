import pandas as pd
from ingestion.readers import missing_summary

#--- Function : test_missing_summary_basic ---
def test_missing_summary_basic():
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": [None, None, 6],
        "c": [7, 8, 9]
    })

    summary = missing_summary(df)

    assert isinstance(summary, pd.DataFrame)
    assert list(summary.columns) == ["missing_count", "missing_percent"]

    assert summary.loc["b", "missing_count"] == 2
    assert summary.loc["a", "missing_count"] == 1
    assert summary.loc["c", "missing_count"] == 0

#--- Function : test_missing_summary_percentages ---
def test_missing_summary_percentages():
    df = pd.DataFrame({
        "x": [None, 1, None, 1]
    })

    summary = missing_summary(df)

    assert summary.loc["x", "missing_percent"] == 50.0

#--- Function : test_missing_summary_sorted_desc ---
def test_missing_summary_sorted_desc():
    df = pd.DataFrame({
        "a": [None, 1],
        "b": [None, None],
        "c": [1, 2]
    })

    summary = missing_summary(df)

    assert summary.index.tolist() == ["b", "a", "c"]

#--- Function : test_missing_summary_no_missing ---
def test_missing_summary_no_missing():
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4]
    })

    summary = missing_summary(df)

    assert (summary["missing_count"] == 0).all()
    assert (summary["missing_percent"] == 0).all()

#--- Function : test_missing_summary_empty_dataframe ---
def test_missing_summary_empty_dataframe():
    df = pd.DataFrame()

    summary = missing_summary(df)

    assert isinstance(summary, pd.DataFrame)
    assert summary.empty
