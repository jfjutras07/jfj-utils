import pandas as pd
import numpy as np
from data_preprocessing.scaling import robust_scaler
from sklearn.preprocessing import RobustScaler

#--- Test: robust_scaler ---
def test_robust_scaler():
    train_df = pd.DataFrame({
        "age": [20, 30, 40, 1000],  # 1000 is outlier
        "income": [2000, 3000, 4000, 50000],  # 50000 is outlier
        "gender": [0, 1, 1, 0],  # binary column should be excluded
        "score_S": [1, 0, 0, 1]  # one-hot dummy column should be excluded
    })

    test_df = pd.DataFrame({
        "age": [25, 35, 500],
        "income": [2500, 3500, 20000],
        "gender": [1, 0, 1],
        "score_S": [0, 1, 0]
    })

    # Test robust scaling
    train_scaled, test_scaled, scaler = robust_scaler(train_df, test_df)
    # Check scaler is instance of RobustScaler
    assert isinstance(scaler, RobustScaler)
    # Binary and one-hot columns remain unchanged
    assert all(train_scaled["gender"].isin([0,1]))
    assert all(test_scaled["score_S"].isin([0,1]))
    # Check that scaled columns are floats
    assert np.issubdtype(train_scaled["age"].dtype, np.floating)
    assert np.issubdtype(train_scaled["income"].dtype, np.floating)
    # Check that test scaled values are within reasonable range
    assert np.min(test_scaled["age"]) < 1.5 and np.max(test_scaled["age"]) > -1.5
    assert np.min(test_scaled["income"]) < 2 and np.max(test_scaled["income"]) > -2

    # Test with columns=None and numeric-only detection
    train_scaled2, test_scaled2, scaler2 = robust_scaler(train_df, test_df, columns=None)
    assert "age" in train_scaled2.columns
    assert "income" in train_scaled2.columns
