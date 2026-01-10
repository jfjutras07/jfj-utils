import pandas as pd
import numpy as np
import io
import sys
from eda.chi_square_test import chi_square_test

#--- Function : test_chi_square_test ---
def test_chi_square_test():
    # Create a sample categorical DataFrame
    df = pd.DataFrame({
        "Gender": ["M", "F", "F", "M", "F", "M", "M", "F", "M", "F"],
        "JobTitle": ["Dev", "Dev", "QA", "QA", "Dev", "QA", "Dev", "QA", "Dev", "QA"]
    })

    # Run chi-square test
    result = chi_square_test(df, col1="Gender", col2="JobTitle")
    
    # Check result is a dictionary
    assert isinstance(result, dict)
    
    # Check all expected keys are present
    for key in ["chi2", "p_value", "dof", "cramers_v"]:
        assert key in result

    # Check types of values
    assert isinstance(result["chi2"], (float, np.floating))
    assert isinstance(result["p_value"], (float, np.floating))
    assert isinstance(result["dof"], int)
    assert isinstance(result["cramers_v"], (float, np.floating))

    # Check optional print of contingency table
    captured_output = io.StringIO()
    sys.stdout = captured_output
    chi_square_test(df, col1="Gender", col2="JobTitle", show_table=True)
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__
    assert "Contingency Table" in output
    assert "Chi-square test" in output
    assert "Cramér's V" in output

    # Edge case: column not in DataFrame
    try:
        chi_square_test(df, col1="Gender", col2="NonExistent")
        assert False  # Should not reach here
    except KeyError as e:
        assert "NonExistent" in str(e)
