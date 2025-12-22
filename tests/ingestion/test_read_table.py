import pandas as pd
from unittest.mock import patch
from ingestion.readers import read_table

#--- Function : test_read_table_file_does_not_exist ---
def test_read_table_file_does_not_exist():
    with patch("os.path.exists", return_value=False):
        df = read_table("fake.csv")

        assert df is None

#--- Function : test_read_table_csv ---
def test_read_table_csv():
    fake_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    with patch("os.path.exists", return_value=True), \
         patch("pandas.read_csv", return_value=fake_df):

        df = read_table("file.csv")

        assert isinstance(df, pd.DataFrame)
        assert df.equals(fake_df)

#--- Function : test_read_table_excel ---
def test_read_table_excel():
    fake_df = pd.DataFrame({"x": [10, 20]})

    with patch("os.path.exists", return_value=True), \
         patch("pandas.read_excel", return_value=fake_df):

        df = read_table("file.xlsx")

        assert isinstance(df, pd.DataFrame)
        assert df.equals(fake_df)

#--- Function : test_read_table_parquet ---
def test_read_table_parquet():
    fake_df = pd.DataFrame({"y": [1, 2, 3]})

    with patch("os.path.exists", return_value=True), \
         patch("pandas.read_parquet", return_value=fake_df):

        df = read_table("file.parquet")

        assert isinstance(df, pd.DataFrame)
        assert df.equals(fake_df)

#--- Function : test_read_table_json ---
def test_read_table_json():
    fake_df = pd.DataFrame({"z": ["a", "b"]})

    with patch("os.path.exists", return_value=True), \
         patch("pandas.read_json", return_value=fake_df):

        df = read_table("file.json")

        assert isinstance(df, pd.DataFrame)
        assert df.equals(fake_df)

#--- Function : test_read_table_unsupported_extension ---
def test_read_table_unsupported_extension():
    with patch("os.path.exists", return_value=True):
        df = read_table("file.txt")

        assert df is None

#--- Function : test_read_table_exception_handled ---
def test_read_table_exception_handled():
    with patch("os.path.exists", return_value=True), \
         patch("pandas.read_csv", side_effect=Exception("boom")):

        df = read_table("file.csv")

        assert df is None
