import pandas as pd
import numpy as np
import statsmodels.api as sm
from modeling.regression import linear_regression

#--- Function : test_linear_regression_basic ---
def test_linear_regression_basic():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    })

    model = linear_regression(df, outcome="y", predictors=["x"])

    assert hasattr(model, "params")
    assert "x" in model.params.index
    assert "Intercept" in model.params.index

#--- Function : test_linear_regression_perfect_fit ---
def test_linear_regression_perfect_fit():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    })

    model = linear_regression(df, outcome="y", predictors=["x"])

    assert np.isclose(model.params["x"], 2.0)
    assert np.isclose(model.rsquared, 1.0)

#--- Function : test_linear_regression_prediction ---
def test_linear_regression_prediction():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    })

    model = linear_regression(df, outcome="y", predictors=["x"])
    X = sm.add_constant(pd.DataFrame({"x": [6]}))
    y_pred = model.predict(X)

    assert np.isclose(y_pred.iloc[0], 12.0)

#--- Function : test_linear_regression_multiple_predictors ---
def test_linear_regression_multiple_predictors():
    df = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5],
        "x2": [5, 4, 3, 2, 1],
        "y":  [3, 4, 5, 6, 7]
    })

    model = linear_regression(df, outcome="y", predictors=["x1", "x2"])

    assert "x1" in model.params.index
    assert "x2" in model.params.index

#--- Function : test_linear_regression_no_predictors_raises ---
def test_linear_regression_no_predictors_raises():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [1, 2, 3]
    })

    try:
        linear_regression(df, outcome="y", predictors=[])
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True
