import pandas as pd
import numpy as np
from modeling.regression_models import linear_regression

#--- Function : test_linear_regression_basic ---
def test_linear_regression_basic():
    train_df = pd.DataFrame({
        "y": np.random.rand(50),
        "x1": np.random.rand(50),
        "x2": np.random.rand(50)
    })

    test_df = pd.DataFrame({
        "y": np.random.rand(20),
        "x1": np.random.rand(20),
        "x2": np.random.rand(20)
    })

    predictors = ["x1", "x2"]

    model = linear_regression(
        train_df=train_df,
        test_df=test_df,
        outcome="y",
        predictors=predictors
    )

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "fit")

#--- Function : test_linear_regression_for_stacking ---
def test_linear_regression_for_stacking():
    train_df = pd.DataFrame({
        "y": np.random.rand(30),
        "x1": np.random.rand(30),
        "x2": np.random.rand(30)
    })

    test_df = pd.DataFrame({
        "y": np.random.rand(10),
        "x1": np.random.rand(10),
        "x2": np.random.rand(10)
    })

    predictors = ["x1", "x2"]

    pipeline = linear_regression(
        train_df=train_df,
        test_df=test_df,
        outcome="y",
        predictors=predictors,
        for_stacking=True
    )

    assert pipeline is not None
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict")
