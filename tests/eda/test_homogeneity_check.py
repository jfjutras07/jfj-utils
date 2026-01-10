import pandas as pd
import numpy as np
import io
import sys
from eda.check_normality import homogeneity_check 

#--- Function : test_homogeneity_check ---
def test_homogeneity_check():
    # Create a sample DataFrame
    df = pd.DataFrame({
        "score": [10, 12, 11, 13, 12, 14, 10, 11, 12, 13],
        "class": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"]
    })

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Run homogeneity check
    result = homogeneity_check(df, value_col="score", group_col="class", center="median")

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Check result type
    assert isinstance(result, dict)

    # Check all expected keys
    expected_keys = [
        'levene_stat', 'levene_p',
        'bartlett_stat', 'bartlett_p',
        'brown_forsythe_stat', 'brown_forsythe_p',
        'fligner_stat', 'fligner_p'
    ]
    for key in expected_keys:
        assert key in result
        assert isinstance(result[key], (float, np.floating))

    # Check that stdout captured print statements
    output = captured_output.getvalue()
    assert "Levene's test" in output
    assert "Bartlett's test" in output
    assert "Brown-Forsythe test" in output
    assert "Fligner-Killeen test" in output

    # Edge case: non-numeric value column
    df_invalid = pd.DataFrame({
        "score": ["a", "b", "c"],
        "group": ["X", "Y", "Z"]
    })
    try:
        homogeneity_check(df_invalid, value_col="score", group_col="group")
        assert False  # Should not reach here
    except ValueError as e:
        assert str(e) == "score must be numeric."
