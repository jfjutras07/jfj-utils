import pandas as pd
import numpy as np
from data_preprocessing.scaling import standard_scaler
from sklearn.preprocessing import StandardScaler

#--- Test: standard_scaler ---
def test_standard_scaler():
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

    # Test standard scaling
    train_scaled, test_scaled, scaler = standard_scaler(train_df, test_df)
    # Check scaler is instance of StandardScaler
    assert isinstance(scaler, StandardScaler)
    # Binary and one-hot columns remain unchanged
    assert all(train_scaled["gender"].isin([0,1]))
    assert all(test_scaled["score_S"].isin([0,1]))
    # Check that scaled columns are floats
    assert np.issubdtype(train_scaled["age"].dtype, np.floating)
    assert np.issubdtype(train_scaled["income"].dtype, np.floating)
    # Mean of train columns approx 0
    assert abs(train_scaled["age"].mean()) < 1e-6
    assert abs(train_scaled["income"].mean()) < 1e-6
    # Variance of train columns approx 1
    assert abs(train_scaled["age"].std() - 1) < 1e-6
    assert abs(train_scaled["income"].std() - 1) < 1e-6

    # Test with columns=None and numeric-only detection
    train_scaled2, test_scaled2, scaler2 = standard_scaler(train_df, test_df, columns=None)
    assert "age" in train_scaled2.columns
    assert "income" in train_scaled2.columns
