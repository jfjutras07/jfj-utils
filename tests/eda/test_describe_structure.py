import pandas as pd
import numpy as np
import io
import sys
from eda.describe_structure import describe_structure

#--- Function : test_describe_structure ---
def test_describe_structure():
    # Sample DataFrame
    df = pd.DataFrame({
        "num_col": [1, 2, 3, 4, 5],
        "bool_col": [True, False, True, False, True],
        "cat_col": ["A", "B", "A", "B", "A"],
        "text_col": ["foo", "bar", "baz", "qux", "quux"],
        "id_col": [101, 102, 103, 104, 105],
        "date_col": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"])
    })

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Run describe_structure
    describe_structure(df, id_cols=["id_col"], date_cols=["date_col"], cat_threshold=10, max_unique_display=5)

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Check that key sections are in the printed output
    output = captured_output.getvalue()
    assert "=== Numeric Columns ===" in output
    assert "=== Boolean Columns ===" in output
    assert "=== Categorical Columns ===" in output
    assert "=== Text Columns ===" in output
    assert "=== ID Columns ===" in output
    assert "=== Date Columns ===" in output

    # Check that some expected values are printed
    assert "num_col" in output
    assert "bool_col" in output
    assert "cat_col" in output
    assert "text_col" in output
    assert "id_col" in output
    assert "date_col" in output

    # Edge case: empty DataFrame
    df_empty = pd.DataFrame()
    captured_output = io.StringIO()
    sys.stdout = captured_output
    describe_structure(df_empty)
    output_empty = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    # Should still print sections headers
    assert "=== Numeric Columns ===" in output_empty
    assert "=== Boolean Columns ===" in output_empty
    assert "=== Categorical Columns ===" in output_empty
    assert "=== Text Columns ===" in output_empty
    assert "=== ID Columns ===" in output_empty
    assert "=== Date Columns ===" in output_empty
