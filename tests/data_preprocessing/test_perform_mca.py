import pandas as pd
from data_preprocessing.dimensionality_reduction import perform_mca
from prince import MCA

#--- Test: perform_mca ---
def test_perform_mca():
    df_train = pd.DataFrame({
        "gender": ["M", "F", "F", "M"],
        "region": ["north", "south", "east", "west"],
        "income_cat": ["low", "medium", "high", "medium"]
    })

    df_test = pd.DataFrame({
        "gender": ["F", "M"],
        "region": ["east", "north"],
        "income_cat": ["medium", "low"]
    })

    # Test MCA without returning model
    train_mca, test_mca, eigen_summary = perform_mca(df_train, df_test, n_components=2, return_model=False)

    # Check types
    assert isinstance(train_mca, pd.DataFrame)
    assert isinstance(test_mca, pd.DataFrame)
    assert isinstance(eigen_summary, pd.DataFrame)

    # Check column names
    assert list(train_mca.columns) == ["MCA_Dim1", "MCA_Dim2"]
    assert list(test_mca.columns) == ["MCA_Dim1", "MCA_Dim2"]

    # Check number of rows
    assert train_mca.shape[0] == df_train.shape[0]
    assert test_mca.shape[0] == df_test.shape[0]

    # Test MCA returning model
    train_mca2, test_mca2, eigen_summary2, model = perform_mca(df_train, df_test, n_components=2, return_model=True)
    assert isinstance(model, MCA)

    # Check that eigenvalues are non-negative
    assert (eigen_summary2["eigenvalue"] >= 0).all()
