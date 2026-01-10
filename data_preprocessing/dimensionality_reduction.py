from prince import FAMD, MCA
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

#--- Function : perform_famd ---
def perform_famd(X_train, X_test=None, n_components=2):
    """
    Perform FAMD and return a structured dictionary of results.
    """
    X_train = X_train.copy()
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[num_cols] = X_train[num_cols].astype(float)
    
    # Fit & Transform
    model = FAMD(n_components=n_components, random_state=42)
    train_coords = model.fit_transform(X_train)
    
    cols = [f'FAMD_Dim{i+1}' for i in range(n_components)]
    df_train_coords = pd.DataFrame(train_coords, columns=cols, index=X_train.index)
    
    df_test_coords = None
    if X_test is not None:
        X_test_aligned = X_test.reindex(columns=X_train.columns).astype({c: float for c in num_cols})
        test_coords = model.transform(X_test_aligned)
        df_test_coords = pd.DataFrame(test_coords, columns=cols, index=X_test.index)
    
    return {
        "coords": df_train_coords,
        "coords_test": df_test_coords,
        "eigenvalues": model.eigenvalues_summary,
        "contributions": model.column_contributions_, # Crucial pour l'interprétabilité
        "model": model
    }

#--- Function : perform_mca ---
def perform_mca(X_train, X_test=None, n_components=2):
    """
    Perform MCA and return a structured dictionary of results.
    """
    model = MCA(n_components=n_components, random_state=42)
    train_coords = model.fit_transform(X_train)
    
    cols = [f'MCA_Dim{i+1}' for i in range(n_components)]
    df_train_coords = pd.DataFrame(train_coords, columns=cols, index=X_train.index)
    
    df_test_coords = None
    if X_test is not None:
        X_test_aligned = X_test.reindex(columns=X_train.columns)
        test_coords = model.transform(X_test_aligned)
        df_test_coords = pd.DataFrame(test_coords, columns=cols, index=X_test.index)
        
    return {
        "coords": df_train_coords,
        "coords_test": df_test_coords,
        "eigenvalues": model.eigenvalues_summary,
        "contributions": model.column_contributions_,
        "model": model
    }

#--- Function : perform_pca ---
def perform_pca(X_train, X_test=None, variance_threshold=0.95):
    """
    Perform PCA and return a structured dictionary of results.
    """
    if X_test is not None:
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    model = PCA(n_components=variance_threshold)
    train_coords = model.fit_transform(X_train)

    cols = [f'PC{i+1}' for i in range(train_coords.shape[1])]
    df_train_coords = pd.DataFrame(train_coords, columns=cols, index=X_train.index)

    df_test_coords = None
    if X_test is not None:
        test_coords = model.transform(X_test)
        df_test_coords = pd.DataFrame(test_coords, columns=cols, index=X_test.index)

    # Création d'un DataFrame de contributions pour la PCA (Loadings)
    # PCA.components_ est (n_composantes, n_features)
    contributions = pd.DataFrame(
        model.components_.T, 
        columns=cols, 
        index=X_train.columns
    )

    return {
        "coords": df_train_coords,
        "coords_test": df_test_coords,
        "eigenvalues": pd.DataFrame({
            "Variance": model.explained_variance_,
            "% Variance": model.explained_variance_ratio_,
            "% Cumulée": np.cumsum(model.explained_variance_ratio_)
        }),
        "contributions": contributions,
        "model": model
    }
