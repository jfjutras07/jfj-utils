import pandas as pd
from jfj_utils.preprocessing.name_cleaning import clean_names

#--- Test: clean_names ---
def test_clean_names():
    df = pd.DataFrame({
        "first_name": ["john", "bob", "O'neil", "anne-marie", "mcdonald"],
        "last_name": ["smith", None, None, "brown", "mcgregor"]
    })
    df_clean = clean_names(df)

    assert df_clean.loc[0,'first_name_clean'].lower() == 'john'
    assert df_clean.loc[0,'last_name_clean'].lower() == 'smith'
    assert df_clean.loc[1,'first_name_clean'].lower() == 'bob'
    assert pd.isna(df_clean.loc[1,'last_name_clean'])
    assert df_clean.loc[2,'first_name_clean'] == "O'Neil"
    assert pd.isna(df_clean.loc[2,'last_name_clean'])
    assert df_clean.loc[3,'first_name_clean'] == "Anne-Marie"
    assert df_clean.loc[3,'last_name_clean'].lower() == "brown"
    assert df_clean.loc[4,'first_name_clean'] == "McDonald"
    assert df_clean.loc[4,'last_name_clean'] == "McGregor"
