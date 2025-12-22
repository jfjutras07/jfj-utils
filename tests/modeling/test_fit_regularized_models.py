import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from modeling.regularization import fit_regularized_models

@pytest.fixture
def dummy_regression_data():
    # Minimal dataset for regression
    np.random.seed(42)
    X = pd.DataFrame({
        "x1": np.random.rand(10),
        "x2": np.random.rand(10),
        "x3": np.random.rand(10)
    })
    y = pd.DataFrame({
        "y1": X["x1"] * 2 + X["x2"] * 3 + np.random.rand(10),
        "y2": X["x2"] - X["x3"] + np.random.rand(10)
    })
    return train_test_split(X, y, test_size=0.5, random_state=42)

def test_fit_regularized_models(dummy_regression_data):
    X_train, X_test, y_train, y_test = dummy_regression_data

    #Call the function
    result = fit_regularized_models(X_train, y_train, X_test, y_test, n_splits=2)

    #Check that the output dictionary has the expected keys
    assert set(result.keys()) == {"models", "scores", "best_model_name", "best_model"}

    #Check that 'models' contains the three regularized models
    assert set(result["models"].keys()) == {"ridge", "lasso", "elasticnet"}

    #Check that 'scores' is a DataFrame and contains the expected columns
    scores = result["scores"]
    assert isinstance(scores, pd.DataFrame)
    assert set(scores.columns) == {"model", "best_params", "rmse", "r2"}

    #Check that best_model_name is one of the models
    assert result["best_model_name"] in ["ridge", "lasso", "elasticnet"]

    #Check that best_model is a fitted object and predictions have correct shape
    best_model = result["best_model"]
    assert hasattr(best_model, "predict")
    y_pred = best_model.predict(X_test)
    assert y_pred.shape[0] == X_test.shape[0]
