import pandas as pd
from data_preprocessing.encoding import one_hot_encode_columns

#--- Test: one_hot_encode_columns ---
def test_one_hot_encode_columns():
    df = pd.DataFrame({
        "color": ["red", "blue", "green", "red"],
        "size": ["S", "M", "L", "S"],
        "price": [10, 15, 20, 10]
    })

    # Test basic one-hot encoding
    df_encoded = one_hot_encode_columns(df, ["color", "size"], drop_first=False)
    assert "color_red" in df_encoded.columns
    assert "color_blue" in df_encoded.columns
    assert "color_green" in df_encoded.columns
    assert "size_S" in df_encoded.columns
    assert "size_M" in df_encoded.columns
    assert "size_L" in df_encoded.columns
    # Original numeric column remains
    assert "price" in df_encoded.columns
    # Check values
    assert df_encoded.loc[0,"color_red"] == 1
    assert df_encoded.loc[1,"color_blue"] == 1
    assert df_encoded.loc[2,"color_green"] == 1
    assert df_encoded.loc[3,"size_S"] == 1

    # Test with drop_first=True
    df_encoded2 = one_hot_encode_columns(df, ["color"], drop_first=True)
    # Check that one column is dropped
    assert "color_red" not in df_encoded2.columns or "color_blue" in df_encoded2.columns

    # Test with train_reference
    train_ref = pd.DataFrame({
        "color": ["red", "blue"],
        "size": ["S", "M"]
    })
    df_new = pd.DataFrame({
        "color": ["green", "red"],
        "size": ["L", "S"]
    })
    df_encoded3 = one_hot_encode_columns(df_new, ["color", "size"], train_reference=train_ref)
    # Ensure all columns from train_reference exist
    for col in ["color_red", "color_blue", "size_S", "size_M"]:
        assert col in df_encoded3.columns
    # New unseen categories should not create new columns
    assert "color_green" not in df_encoded3.columns
