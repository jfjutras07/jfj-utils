import pandas as pd
import numpy as np
from data_preprocessing.pca import perform_pca

#--- Function : test_perform_pca_basic ---
def test_perform_pca_basic():
    X = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [2, 3, 4, 5],
        "c": [5, 6, 7, 8]
    })

    df_pca, var_ratio = perform_pca(X, variance_threshold=0.9)

    assert isinstance(df_pca, pd.DataFrame)
    assert df_pca.shape[0] == X.shape[0]
    assert len(var_ratio) == df_pca.shape[1]

#--- Function : test_perform_pca_column_names ---
def test_perform_pca_column_names():
    X = np.random.rand(10, 4)

    df_pca, _ = perform_pca(X, variance_threshold=0.95)

    assert all(col.startswith("PC") for col in df_pca.columns)

#--- Function : test_perform_pca_variance_threshold_low ---
def test_perform_pca_variance_threshold_low():
    X = np.random.rand(20, 5)

    df_pca, var_ratio = perform_pca(X, variance_threshold=0.5)

    assert df_pca.shape[1] >= 1
    assert sum(var_ratio) >= 0.5

#--- Function : test_perform_pca_returns_variance_ratio ---
def test_perform_pca_returns_variance_ratio():
    X = np.random.rand(15, 3)

    _, var_ratio = perform_pca(X)

    assert isinstance(var_ratio, (list, tuple, np.ndarray))
    assert all(v >= 0 for v in var_ratio)
