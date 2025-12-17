import pandas as pd
from jfj_utils.preprocessing.name_cleaning import clean_names

#--- Function test_clean_names ---
def test_clean_names():
    df = pd.DataFrame({
        "first_name": [" john", "ANNE-MARIE", "o'neil", None, "McDonald"],
        "last_name": ["doe", "smith", None, "brown", "mcgregor"]
    })

    df_clean = clean_names(df)

    assert df_clean.loc[0, 'first_name_clean'] == "John"
    assert df_clean.loc[0, 'last_name_clean'] == "Doe"

    assert df_clean.loc[1, 'first_name_clean'] == "Anne-Marie"
    assert df_clean.loc[1, 'last_name_clean'] == "Smith"

    assert df_clean.loc[2, 'first_name_clean'] == "O'Neil"
    assert pd.isna(df_clean.loc[2, 'last_name_clean'])

    assert pd.isna(df_clean.loc[3, 'first_name_clean'])
    assert df_clean.loc[3, 'last_name_clean'] == "Brown"

    assert df_clean.loc[4, 'first_name_clean'] == "McDonald"
    assert df_clean.loc[4, 'last_name_clean'] == "McGregor"
