import pandas as pd
import numpy as np
from data_preprocessing.famd import perform_famd
from prince import FAMD

#--- Test: perform_famd ---
def test_perform_famd():
    df_train = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "income": [2000, 3000, 4000, 5000],
        "gender": ["M", "F", "F", "M"],
        "region": ["north", "south", "east", "west"]
    })

    df_test = pd.DataFrame({
        "age": [25, 35],
        "income": [2500, 3500],
        "gender": ["F", "M"],
        "region": ["east", "north"]
    })

    # Test FAMD without returning model
    train_famd, test_famd, eigen_summary = perform_famd(df_train, df_test, n_components=2, return_model=False)

    # Check types
    assert isinstance(train_famd, pd.DataFrame)
    assert isinstance(test_famd, pd.DataFrame)
    assert isinstance(eigen_summary, pd.DataFrame)

    # Check column names
    assert list(train_famd.columns) == ["FAMD_Dim1", "FAMD_Dim2"]
    assert list(test_famd.columns) == ["FAMD_Dim1", "FAMD_Dim2"]

    # Check number of rows
    assert train_famd.shape[0] == df_train.shape[0]
    assert test_famd.shape[0] == df_test.shape[0]

    # Test FAMD returning model
    train_famd2, test_famd2, eigen_summary2, model = perform_famd(df_train, df_test, n_components=2, return_model=True)
    assert isinstance(model, FAMD)

    # Check that eigenvalues are non-negative
    assert (eigen_summary2["eigenvalue"] >= 0).all()
