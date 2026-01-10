import pandas as pd
from data_preprocessing.encoding import label_encode_columns

#--- Test: label_encode_columns ---
def test_label_encode_columns():
    df = pd.DataFrame({
        "color": ["red", "blue", "green", "red"],
        "size": ["S", "M", "L", "S"],
        "price": [10, 15, 20, 10]
    })

    # Test basic label encoding
    df_encoded = label_encode_columns(df, ["color", "size"])
    # Check that encoded values are integers
    assert df_encoded["color"].dtype == int
    assert df_encoded["size"].dtype == int
    # Original numeric column remains
    assert "price" in df_encoded.columns

    # Test multiple datasets
    df2 = pd.DataFrame({
        "color": ["green", "red"],
        "size": ["L", "S"],
        "price": [12, 18]
    })
    dfs_encoded = label_encode_columns([df, df2], ["color", "size"])
    assert dfs_encoded[1]["color"].dtype == int
    assert dfs_encoded[1]["size"].dtype == int

    # Test train_reference to enforce mapping
    train_ref = pd.DataFrame({
        "color": ["red", "blue"],
        "size": ["S", "M"]
    })
    df_new = pd.DataFrame({
        "color": ["red", "green"],  # "green" is unseen
        "size": ["S", "L"]          # "L" is unseen
    })
    df_encoded2 = label_encode_columns(df_new, ["color", "size"], train_reference=train_ref)
    # Known values follow train mapping
    assert df_encoded2.loc[0,"color"] >= 0
    assert df_encoded2.loc[0,"size"] >= 0
    # Unknown values mapped to -1
    assert df_encoded2.loc[1,"color"] == -1
    assert df_encoded2.loc[1,"size"] == -1
