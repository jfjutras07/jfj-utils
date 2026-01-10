import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from modeling.regularization import compare_regularized_models

#--- Function : test_compare_regularized_models_basic ---
def test_compare_regularized_models_basic(monkeypatch):
    train_df = pd.DataFrame({
        "y": np.random.rand(40),
        "x1": np.random.rand(40),
        "x2": np.random.rand(40)
    })

    test_df = pd.DataFrame({
        "y": np.random.rand(15),
        "x1": np.random.rand(15),
        "x2": np.random.rand(15)
    })

    predictors = ["x1", "x2"]

    # Dummy regression result factory
    def dummy_regularized(*args, **kwargs):
        model = LinearRegression()
        model.fit(train_df[predictors], train_df["y"])
        return {
            "model": model,
            "metrics": {
                "R2": 0.5,
                "MAE": 0.1,
                "RMSE": 0.2
            },
            "coefficients": pd.DataFrame({
                "Feature": predictors,
                "Coefficient": [0.5, 0.0]
            })
        }

    # Monkeypatch all regularized regressions
    monkeypatch.setattr("modeling.regularization.lasso_regression", dummy_regularized)
    monkeypatch.setattr("modeling.regularization.ridge_regression", dummy_regularized)
    monkeypatch.setattr("modeling.regularization.elasticnet_regression", dummy_regularized)

    winner_model = compare_regularized_models(
        train_df=train_df,
        test_df=test_df,
        outcome="y",
        predictors=predictors,
        cv=3
    )

    assert winner_model is not None
    assert hasattr(winner_model, "predict")
    assert hasattr(winner_model, "fit")
