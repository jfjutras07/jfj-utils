import pandas as pd
from data_preprocessing.normalizing import normalize_columns

#--- Test: normalize_columns ---
def test_normalize_columns():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "income": [2000, 3000, 4000, 5000],
        "gender": [0, 1, 1, 0],
        "score_S": [1, 0, 0, 1]
    })

    # Standard scaling
    df_scaled, _ = normalize_columns(df, method="standard")
    assert abs(df_scaled["age"].mean()) < 1e-6
    assert abs(df_scaled["income"].mean()) < 1e-6
    assert all(df_scaled["gender"].isin([0, 1]))
    assert all(df_scaled["score_S"].isin([0, 1]))

    # MinMax scaling
    df_scaled2, _ = normalize_columns(df, method="minmax")
    assert df_scaled2["age"].min() == 0
    assert df_scaled2["age"].max() == 1

    # Robust scaling
    df_scaled3, _ = normalize_columns(df, method="robust")
    assert df_scaled3["age"].median() == 0

    # Multiple datasets
    df2 = pd.DataFrame({"age": [25, 35], "income": [2500, 3500], "gender": [1, 0]})
    dfs_scaled, _ = normalize_columns([df, df2], method="standard")
    assert len(dfs_scaled) == 2
    assert dfs_scaled[1]["age"].dtype == float

    # Invalid method
    try:
        normalize_columns(df, method="unknown")
        assert False
    except ValueError:
        assert True
