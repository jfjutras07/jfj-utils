import pandas as pd
from data_preprocessing.encoding import binary_encode_columns  # adjust to your module

#--- Test: binary_encode_columns ---
def test_binary_encode_columns():
    df = pd.DataFrame({
        "gender": ["M", "F", "F", "M"],
        "smoker": ["yes", "no", "no", "yes"],
        "age": [25, 30, 22, 40]
    })

    binary_map = {
        "gender": {"M":1, "F":0},
        "smoker": {"yes":1, "no":0}
    }

    # Test basic binary encoding
    df_encoded = binary_encode_columns(df, binary_map)
    assert df_encoded.loc[0,"gender"] == 1
    assert df_encoded.loc[1,"gender"] == 0
    assert df_encoded.loc[0,"smoker"] == 1
    assert df_encoded.loc[1,"smoker"] == 0
    # Original numeric column remains
    assert "age" in df_encoded.columns

    # Test multiple datasets input
    df2 = pd.DataFrame({
        "gender": ["F", "M"],
        "smoker": ["no", "yes"],
        "age": [28, 35]
    })
    dfs_encoded = binary_encode_columns([df, df2], binary_map)
    assert dfs_encoded[1].loc[0,"gender"] == 0
    assert dfs_encoded[1].loc[1,"smoker"] == 1

    # Test train_reference to enforce mappings
    train_ref = pd.DataFrame({
        "gender": ["M", "F"],
        "smoker": ["yes", "no"]
    })
    df_new = pd.DataFrame({
        "gender": ["M", "F"],
        "smoker": ["no", "yes"]
    })
    df_encoded2 = binary_encode_columns(df_new, binary_map, train_reference=train_ref)
    assert df_encoded2.loc[0,"gender"] == 1
    assert df_encoded2.loc[0,"smoker"] == 0

    # Test strict mode raises ValueError for invalid mapping
    df_invalid = pd.DataFrame({"gender": ["X"], "smoker": ["yes"]})
    try:
        binary_encode_columns(df_invalid, binary_map)
        assert False  # Should not reach here
    except ValueError:
        assert True
