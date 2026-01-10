import pandas as pd
import numpy as np
from eda.multicollinearity import VIF_check

#--- Function : test_VIF_check ---
def test_VIF_check():
    # Basic numeric DataFrame
    df = pd.DataFrame({
        "x1": np.random.rand(10),
        "x2": np.random.rand(10),
        "x3": np.random.rand(10)
    })

    # Should run without error
    result = VIF_check(df)
    assert result is None  # Function prints VIF but returns None

    # DataFrame with categorical column
    df_cat = pd.DataFrame({
        "num1": np.random.rand(10),
        "cat1": ["a", "b"] * 5
    })
    result = VIF_check(df_cat)
    assert result is None  # Should handle categorical automatically

    # DataFrame with constant column
    df_const = pd.DataFrame({
        "x1": np.random.rand(10),
        "x2": np.ones(10)  # Constant column should be removed
    })
    result = VIF_check(df_const)
    assert result is None  # Should ignore constant column

    # DataFrame with too few valid columns
    df_few = pd.DataFrame({
        "x1": np.ones(5)
    })
    try:
        VIF_check(df_few)
        assert False  # Should not reach here
    except ValueError as e:
        assert str(e) == "Not enough valid predictors to compute VIF."
