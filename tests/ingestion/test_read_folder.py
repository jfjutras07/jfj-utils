import pandas as pd
from unittest.mock import patch
from ingestion.readers import read_folder

#--- Function : test_read_folder_folder_does_not_exist ---
def test_read_folder_folder_does_not_exist():
    with patch("os.path.exists", return_value=False):
        dfs = read_folder("fake_folder")

        assert dfs == {}

#--- Function : test_read_folder_loads_supported_files ---
def test_read_folder_loads_supported_files():
    fake_df = pd.DataFrame({"a": [1]})

    with patch("os.path.exists", return_value=True), \
         patch("os.listdir", return_value=["file1.csv", "file2.xlsx"]), \
         patch("ingestion.readers.read_table", return_value=fake_df) as mock_read:

        dfs = read_folder("my_folder")

        assert isinstance(dfs, dict)
        assert "file1" in dfs
        assert "file2" in dfs
        assert dfs["file1"].equals(fake_df)
        assert dfs["file2"].equals(fake_df)
        assert mock_read.call_count == 2

#--- Function : test_read_folder_filters_file_types ---
def test_read_folder_filters_file_types():
    fake_df = pd.DataFrame({"a": [1]})

    with patch("os.path.exists", return_value=True), \
         patch("os.listdir", return_value=["file.csv", "file.xlsx"]), \
         patch("ingestion.readers.read_table", return_value=fake_df):

        dfs = read_folder("my_folder", file_types=[".csv"])

        assert "file" in dfs
        assert len(dfs) == 1

#--- Function : test_read_folder_ignores_unsupported_files ---
def test_read_folder_ignores_unsupported_files():
    with patch("os.path.exists", return_value=True), \
         patch("os.listdir", return_value=["file.txt", "file.docx"]), \
         patch("ingestion.readers.read_table") as mock_read:

        dfs = read_folder("my_folder")

        assert dfs == {}
        mock_read.assert_not_called()

#--- Function : test_read_folder_read_table_returns_none ---
def test_read_folder_read_table_returns_none():
    with patch("os.path.exists", return_value=True), \
         patch("os.listdir", return_value=["file.csv"]), \
         patch("ingestion.readers.read_table", return_value=None):

        dfs = read_folder("my_folder")

        assert dfs == {}
