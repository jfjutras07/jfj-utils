import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from modeling.regression_stacking import stacking_ensemble

#--- Function : test_stacking_ensemble_basic ---
def test_stacking_ensemble_basic():
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

    base_estimators = [
        ("lr1", LinearRegression()),
        ("lr2", LinearRegression())
    ]

    model = stacking_ensemble(
        train_df=train_df,
        test_df=test_df,
        outcome="y",
        predictors=predictors,
        base_estimators=base_estimators,
        final_estimator=Ridge(),
        cv=3
    )

    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
