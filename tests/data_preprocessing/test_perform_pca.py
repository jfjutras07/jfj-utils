import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from data_preprocessing.dimensionality_reduction import perform_pca

#--- Test: perform_pca ---
def test_perform_pca():
    # Sample train dataset
    df_train = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "income": [2000, 3000, 4000, 5000],
        "score": [1, 3, 5, 7]
    })

    # Sample test dataset
    df_test = pd.DataFrame({
        "age": [25, 35],
        "income": [2500, 3500],
        "score": [2, 4]
    })

    # Execute PCA
    results = perform_pca(df_train, df_test, variance_threshold=0.95)
    
    train_pca = results["coords"]
    test_pca = results["coords_test"]
    eigen_summary = results["eigenvalues"]
    model = results["model"]

    # Check types
    assert isinstance(train_pca, pd.DataFrame)
    assert isinstance(test_pca, pd.DataFrame)
    assert isinstance(eigen_summary, pd.DataFrame)
    assert isinstance(model, PCA)

    # Check column names
    expected_cols = [f'PC{i+1}' for i in range(train_pca.shape[1])]
    assert list(train_pca.columns) == expected_cols
    assert list(test_pca.columns) == expected_cols

    # Check number of rows
    assert train_pca.shape[0] == df_train.shape[0]
    assert test_pca.shape[0] == df_test.shape[0]

    # Check explained variance is non-negative
    # (Note: Using the 'eigenvalue' column created in the function)
    assert (eigen_summary["eigenvalue"] >= 0).all()
