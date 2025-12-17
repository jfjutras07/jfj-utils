import pandas as pd
from jfj_utils.preprocessing.name_cleaning import clean_names_multiple

#--- Test: clean_names_multiple ---
def test_clean_names_multiple():
    dfs = {
        "file1": pd.DataFrame({
            "first_name": ["john","bob"],
            "last_name": ["smith", None]
        }),
        "file2": pd.DataFrame({
            "first_name": ["o'connor","mary-jane"],
            "last_name": [None,"brown"]
        })
    }
    dfs_clean = clean_names_multiple(dfs)
    df1 = dfs_clean["file1"]
    df2 = dfs_clean["file2"]

    #Check file1
    assert df1.loc[0,'first_name'].lower() == "john"
    assert df1.loc[0,'last_name'].lower() == "smith"
    assert df1.loc[1,'first_name'].lower() == "bob"
    assert pd.isna(df1.loc[1,'last_name'])

    #Check file2
    assert df2.loc[0,'first_name'].lower() == "o'connor"
    assert pd.isna(df2.loc[0,'last_name'])
    assert df2.loc[1,'first_name'].lower() == "mary-jane"
    assert df2.loc[1,'last_name'].lower() == "brown"
