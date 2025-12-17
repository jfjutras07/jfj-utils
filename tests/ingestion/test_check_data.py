import pandas as pd
from unittest.mock import patch
from ingestion.readers import check_data

#--- Function : test_check_data_with_dataframe ---
def test_check_data_with_dataframe():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    with patch("jfj_utils.ingestion.readers._check_single_df") as mock_check:
        check_data(df, n=3)

        mock_check.assert_called_once_with(df, 3)

#--- Function : test_check_data_with_dict_of_dataframes ---
def test_check_data_with_dict_of_dataframes():
    dfs = {
        "train": pd.DataFrame({"x": [1, 2]}),
        "test": pd.DataFrame({"x": [3, 4]})
    }

    with patch("jfj_utils.ingestion.readers._check_single_df") as mock_check:
        check_data(dfs, n=2)

        assert mock_check.call_count == 2
        mock_check.assert_any_call(dfs["train"], 2)
        mock_check.assert_any_call(dfs["test"], 2)

#--- Function : test_check_data_with_empty_dict ---
def test_check_data_with_empty_dict():
    with patch("jfj_utils.ingestion.readers._check_single_df") as mock_check:
        check_data({})

        mock_check.assert_not_called()

