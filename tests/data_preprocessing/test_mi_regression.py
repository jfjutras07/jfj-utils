import pandas as pd
import numpy as np
from data_preprocessing.feature_engineering import mi_regression

#--- Function : test_mi_regression_basic ---
def test_mi_regression_basic():
    X = pd.DataFrame({
        "num1": np.random.rand(50),
        "num2": np.random.rand(50),
        "cat1": np.random.choice(["A", "B", "C"], size=50)
    })

    y = pd.Series(np.random.rand(50))

    mi_scores = mi_regression(X, y)

    assert isinstance(mi_scores, pd.Series)
    assert mi_scores.shape[0] == X.shape[1]
    assert set(mi_scores.index) == set(X.columns)
    assert mi_scores.name == "MI Scores"

#--- Function : test_mi_regression_sorted_descending ---
def test_mi_regression_sorted_descending():
    X = pd.DataFrame({
        "x1": np.random.rand(30),
        "x2": np.random.rand(30)
    })

    y = pd.Series(np.random.rand(30))

    mi_scores = mi_regression(X, y)

    assert mi_scores.values.tolist() == sorted(mi_scores.values.tolist(), reverse=True)

#--- Function : test_mi_regression_categorical_handling ---
def test_mi_regression_categorical_handling():
    X = pd.DataFrame({
        "cat": ["low", "medium", "high", "low", "medium", "high"],
        "num": [1, 2, 3, 1, 2, 3]
    })

    y = pd.Series([10.5, 20.1, 15.2, 11.3, 19.8, 14.7])

    mi_scores = mi_regression(X, y)

    assert "cat" in mi_scores.index
    assert "num" in mi_scores.index
    assert mi_scores.isna().sum() == 0
