import pandas as pd
from data_preprocessing.encoding import ordinal_encode_columns

#--- Test: ordinal_encode_columns ---
def test_ordinal_encode_columns():
    df = pd.DataFrame({
        "size": ["S", "M", "L", "S"],
        "quality": ["low", "medium", "high", "medium"],
        "price": [10, 15, 20, 10]
    })

    ordinal_map = {
        "size": ["S", "M", "L"],
        "quality": ["low", "medium", "high"]
    }

    # Test basic ordinal encoding
    df_encoded = ordinal_encode_columns(df, ordinal_map)
    # Check that encoded values are integers
    assert df_encoded["size"].dtype == int
    assert df_encoded["quality"].dtype == int
    # Original numeric column remains
    assert "price" in df_encoded.columns
    # Check correct mapping
    assert df_encoded.loc[0,"size"] == 0
    assert df_encoded.loc[1,"size"] == 1
    assert df_encoded.loc[2,"size"] == 2
    assert df_encoded.loc[0,"quality"] == 0
    assert df_encoded.loc[1,"quality"] == 1
    assert df_encoded.loc[2,"quality"] == 2

    # Test multiple datasets
    df2 = pd.DataFrame({
        "size": ["L", "S"],
        "quality": ["high", "low"],
        "price": [12, 18]
    })
    dfs_encoded = ordinal_encode_columns([df, df2], ordinal_map)
    assert dfs_encoded[1]["size"].dtype == int
    assert dfs_encoded[1]["quality"].dtype == int

    # Test invalid values raise ValueError
    df_invalid = pd.DataFrame({
        "size": ["XL"],  # not in mapping
        "quality": ["medium"],
        "price": [20]
    })
    try:
        ordinal_encode_columns(df_invalid, ordinal_map)
        assert False  # Should not reach here
    except ValueError:
        assert True
