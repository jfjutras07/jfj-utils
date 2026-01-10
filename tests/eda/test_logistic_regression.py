import pandas as pd
import numpy as np
import io
import sys
import statsmodels.api as sm
from eda.classification import logistic_regression

#--- Function : test_logistic_regression ---
def test_logistic_regression():
    # Sample DataFrame
    df = pd.DataFrame({
        "y": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        "x1": np.random.rand(10),
        "x2": np.random.rand(10)
    })

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Run logistic regression
    model = logistic_regression(df, outcome="y", predictors=["x1", "x2"])

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Check return type
    import statsmodels.discrete.discrete_model as dm
    assert isinstance(model, dm.BinaryResultsWrapper)

    # Check that summary was printed
    output = captured_output.getvalue()
    assert "--- Logistic Regression Summary ---" in output
    assert "Logit" in output or "y ~" in output

    # Edge case: no predictors
    try:
        logistic_regression(df, outcome="y", predictors=[])
        assert False  # Should not reach here
    except ValueError as e:
        assert str(e) == "At least one predictor must be provided."
