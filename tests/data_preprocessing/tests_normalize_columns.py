import pandas as pd
from data_preprocessing.scaling import normalize_columns

#--- Test: normalize_columns ---
def test_normalize_columns():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "income": [2000, 3000, 4000, 5000],
        "gender": [0, 1, 1, 0],  # binary column should be excluded
        "score_S": [1, 0, 0, 1]  # one-hot dummy column should be excluded
    })

    # Test standard scaling
    df_scaled, scaler = normalize_columns(df, method="standard")
    assert abs(df_scaled["age"].mean()) < 1e-6
    assert abs(df_scaled["income"].mean()) < 1e-6
    # Binary and one-hot columns remain unchanged
    assert all(df_scaled["gender"].isin([0, 1]))
    assert all(df_scaled["score_S"].isin([0, 1]))

    # Test minmax scaling
    df_scaled2, scaler2 = normalize_columns(df, method="minmax")
    assert df_scaled2["age"].min() == 0
    assert df_scaled2["age"].max() == 1
    assert df_scaled2["income"].min() == 0
    assert df_scaled2["income"].max() == 1

    # Test robust scaling
    df_scaled3, scaler3 = normalize_columns(df, method="robust")
    assert df_scaled3["age"].median() == 0
    assert df_scaled3["income"].median() == 0

    # Test multiple datasets
    df2 = pd.DataFrame({
        "age": [25, 35],
        "income": [2500, 3500],
        "gender": [1, 0]
    })
    dfs_scaled, scalers = normalize_columns([df, df2], method="standard")
    assert len(dfs_scaled) == 2
    assert all(dfs_scaled[1]["age"].dtype == float)
    assert all(dfs_scaled[1]["income"].dtype == float)

    # Test invalid method raises ValueError
    try:
        normalize_columns(df, method="unknown")
        assert False  # should not reach here
    except ValueError:
        assert True
