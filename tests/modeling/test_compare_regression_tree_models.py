import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from modeling.regression_trees import compare_tree_models

#--- Function : test_compare_tree_models_basic ---
def test_compare_tree_models_basic(monkeypatch):
    # Dummy dataset
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

    # Dummy model factory
    def dummy_model(*args, **kwargs):
        model = LinearRegression()
        model.fit(train_df[predictors], train_df["y"])
        return model

    # Monkeypatch all model calls
    monkeypatch.setattr("modeling.regression_trees.catboost_regression", dummy_model)
    monkeypatch.setattr("modeling.regression_trees.decision_tree_regression", dummy_model)
    monkeypatch.setattr("modeling.regression_trees.knn_regression", dummy_model)
    monkeypatch.setattr("modeling.regression_trees.lightgbm_regression", dummy_model)
    monkeypatch.setattr("modeling.regression_trees.random_forest_regression", dummy_model)
    monkeypatch.setattr("modeling.regression_trees.xgboost_regression", dummy_model)

    # Run comparison
    winner = compare_tree_models(
        train_df=train_df,
        test_df=test_df,
        outcome="y",
        predictors=predictors,
        cv=3
    )

    # Assertions
    assert winner is not None
    assert hasattr(winner, "predict")
