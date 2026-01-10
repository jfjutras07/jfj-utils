import pandas as pd
import numpy as np
import io
import sys
import statsmodels.api as sm
import statsmodels.formula.api as smf
from eda.regression import linear_regression

#--- Function : test_linear_regression ---
def test_linear_regression():
    # Sample DataFrame
    df = pd.DataFrame({
        "y": np.random.rand(10),
        "x1": np.random.rand(10),
        "x2": ["A", "B"] * 5
    })
    df["x2"] = df["x2"].astype("category")

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Run linear regression without interactions
    model = linear_regression(df, outcome="y", predictors=["x1", "x2"], include_interactions=False)

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Check return type
    import statsmodels.regression.linear_model as lm
    assert isinstance(model, lm.RegressionResultsWrapper)

    # Check formula contains predictors
    formula_str = str(model.model.formula)
    assert "x1" in formula_str
    assert "C(x2)" in formula_str
    assert "*" not in formula_str  # No interactions

    # Run linear regression with interactions
    captured_output = io.StringIO()
    sys.stdout = captured_output
    model_int = linear_regression(df, outcome="y", predictors=["x1", "x2"], include_interactions=True)
    sys.stdout = sys.__stdout__
    formula_str_int = str(model_int.model.formula)
    assert "*" in formula_str_int  # Interactions included

    # Edge case: no predictors
    try:
        linear_regression(df, outcome="y", predictors=[])
        assert False  # Should not reach here
    except ValueError as e:
        assert str(e) == "At least one predictor must be provided."
