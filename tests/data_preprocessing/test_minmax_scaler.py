import pandas as pd
import numpy as np
from data_preprocessing.scaling import minmax_scaler
from sklearn.preprocessing import MinMaxScaler

#--- Test: minmax_scaler ---
def test_minmax_scaler():
    train_df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "income": [2000, 3000, 4000, 5000],
        "gender": [0, 1, 1, 0],  # binary column should be excluded
        "score_S": [1, 0, 0, 1]  # one-hot dummy column should be excluded
    })

    test_df = pd.DataFrame({
        "age": [25, 35],
        "income": [2500, 3500],
        "gender": [1, 0],
        "score_S": [0, 1]
    })

    # Test minmax scaling
    train_scaled, test_scaled, scaler = minmax_scaler(train_df, test_df)
    # Check that scaled values are in [0,1] for train
    assert train_scaled["age"].min() == 0
    assert train_scaled["age"].max() == 1
    assert train_scaled["income"].min() == 0
    assert train_scaled["income"].max() == 1
    # Test dataset scaled using same scaler
    assert test_scaled["age"].min() >= 0
    assert test_scaled["age"].max() <= 1
    assert test_scaled["income"].min() >= 0
    assert test_scaled["income"].max() <= 1
    # Binary and one-hot columns remain unchanged
    assert all(train_scaled["gender"].isin([0,1]))
    assert all(test_scaled["score_S"].isin([0,1]))
    # Check scaler is instance of MinMaxScaler
    assert isinstance(scaler, MinMaxScaler)

    # Test with columns=None and numeric-only detection
    train_scaled2, test_scaled2, scaler2 = minmax_scaler(train_df, test_df, columns=None)
    assert "age" in train_scaled2.columns
    assert "income" in train_scaled2.columns
