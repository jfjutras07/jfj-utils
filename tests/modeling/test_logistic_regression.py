import pandas as pd
import numpy as np
import statsmodels.api as sm
from modeling.classification import logistic_regression

#--- Function : test_logistic_regression_basic ---
def test_logistic_regression_basic():
    df = pd.DataFrame({
        "x": [0, 1, 2, 3, 4, 5],
        "y": [0, 0, 0, 1, 1, 1]
    })

    model = logistic_regression(df, outcome="y", predictors=["x"])

    assert hasattr(model, "params")
    assert "x" in model.params.index
    assert "Intercept" in model.params.index

#--- Function : test_logistic_regression_predict_proba_range ---
def test_logistic_regression_predict_proba_range():
    df = pd.DataFrame({
        "x": [0, 1, 2, 3, 4, 5],
        "y": [0, 0, 0, 1, 1, 1]
    })

    model = logistic_regression(df, outcome="y", predictors=["x"])
    X = sm.add_constant(df[["x"]])
    probs = model.predict(X)

    assert np.all(probs >= 0)
    assert np.all(probs <= 1)

#--- Function : test_logistic_regression_coefficient_sign ---
def test_logistic_regression_coefficient_sign():
    df = pd.DataFrame({
        "x": [0, 1, 2, 3, 4, 5],
        "y": [0, 0, 0, 1, 1, 1]
    })

    model = logistic_regression(df, outcome="y", predictors=["x"])

    # x increases probability of y=1
    assert model.params["x"] > 0

#--- Function : test_logistic_regression_multiple_predictors ---
def test_logistic_regression_multiple_predictors():
    df = pd.DataFrame({
        "x1": [0, 1, 2, 3, 4, 5],
        "x2": [5, 4, 3, 2, 1, 0],
        "y":  [0, 0, 0, 1, 1, 1]
    })

    model = logistic_regression(df, outcome="y", predictors=["x1", "x2"])

    assert "x1" in model.params.index
    assert "x2" in model.params.index

#--- Function : test_logistic_regression_no_predictors_raises ---
def test_logistic_regression_no_predictors_raises():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [0, 1, 1]
    })

    try:
        logistic_regression(df, outcome="y", predictors=[])
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True
