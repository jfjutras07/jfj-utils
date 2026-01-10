from prince import FAMD
from prince import MCA
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

#--- Function : perform_famd ---
def perform_famd(X_train, X_test=None, n_components=2, return_model=False):
    """
    Perform Factor Analysis of Mixed Data (FAMD) on mixed categorical and numerical variables.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Raw mixed train data (categorical + numerical, NOT scaled).
    X_test : pd.DataFrame, optional
        Raw mixed test data.
    n_components : int, default=2
        Number of latent dimensions to extract.
    return_model : bool, default=False
        Whether to return the fitted FAMD model.
    """
    # Ensure proper data types to prevent prince detection errors
    X_train = X_train.copy()
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[num_cols] = X_train[num_cols].astype(float)
    
    # Initialize FAMD (Prince library)
    famd = FAMD(n_components=n_components, random_state=42)
    
    # Fit and transform on Train
    X_train_famd = famd.fit_transform(X_train)
    
    # Create descriptive column names
    famd_cols = [f'FAMD_Dim{i+1}' for i in range(n_components)]
    
    # Convert to DataFrame
    df_train_famd = pd.DataFrame(X_train_famd, columns=famd_cols, index=X_train.index)
    
    # Handle Test set if provided
    df_test_famd = None
    if X_test is not None:
        X_test = X_test.copy()
        X_test[num_cols] = X_test[num_cols].astype(float)
        X_test_aligned = X_test.reindex(columns=X_train.columns)
        X_test_famd = famd.transform(X_test_aligned)
        df_test_famd = pd.DataFrame(X_test_famd, columns=famd_cols, index=X_test.index)
    
    # Return results
    outputs = [df_train_famd]
    if df_test_famd is not None:
        outputs.append(df_test_famd)
    
    outputs.append(famd.eigenvalues_summary)
    
    if return_model:
        outputs.append(famd)
        
    return tuple(outputs)
    
#--- Function : perform_mca ---
def perform_mca(X_train, X_test=None, n_components=2, return_model=False):
    """
    Perform Multiple Correspondence Analysis (MCA) on categorical/ordinal groups.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Raw categorical/ordinal train data (NOT scaled).
    X_test : pd.DataFrame, optional
        Raw categorical/ordinal test data.
    n_components : int, default=2
        Number of latent dimensions to extract.
    return_model : bool, default=False
        Whether to return the fitted MCA model.
    """
    
    #Initialize MCA (Prince library)
    mca = MCA(n_components=n_components, random_state=42)
    
    #Fit and transform on Train
    X_train_mca = mca.fit_transform(X_train)
    
    #Create descriptive column names
    mca_cols = [f'MCA_Dim{i+1}' for i in range(n_components)]
    
    #Convert to DataFrame
    df_train_mca = pd.DataFrame(X_train_mca, columns=mca_cols, index=X_train.index)
    
    #Handle Test set if provided
    df_test_mca = None
    if X_test is not None:
        # Align columns to ensure the same categories are present
        X_test_aligned = X_test.reindex(columns=X_train.columns)
        X_test_mca = mca.transform(X_test_aligned)
        df_test_mca = pd.DataFrame(X_test_mca, columns=mca_cols, index=X_test.index)
    
    #Return results (flexible return logic)
    outputs = [df_train_mca]
    if df_test_mca is not None:
        outputs.append(df_test_mca)
    
    #Use explained_inertia_ (MCA version of explained_variance_ratio_)
    outputs.append(mca.eigenvalues_summary) 
    
    if return_model:
        outputs.append(mca)
        
    return tuple(outputs)

#--- Function : perform_pca ---
def perform_pca(X_train, X_test=None, variance_threshold=0.95, return_model=False):
    """
    Perform PCA on numeric columns of a train dataset and optionally transform a test dataset.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Scaled numeric train data
    X_test : pd.DataFrame, optional
        Scaled numeric test data (columns will be aligned automatically)
    variance_threshold : float, default=0.95
        Fraction of total variance to retain (determines number of components)
    return_model : bool, default=False
        Whether to return the fitted PCA model (needed to transform test sets)
    
    Returns:
    --------
    df_train_pca : pd.DataFrame
        Train DataFrame with principal components
    df_test_pca : pd.DataFrame, optional
        Test DataFrame with principal components (returned only if X_test is provided)
    explained_variance_ratio : np.array
        Variance explained by each principal component
    pca_model : sklearn.decomposition.PCA, optional
        Fitted PCA model (returned only if return_model=True)
    """

    #Align train/test columns if test provided
    if X_test is not None:
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    #Fit PCA on train
    pca = PCA(n_components=variance_threshold)
    X_train_pca = pca.fit_transform(X_train)

    #Create column names
    pca_cols = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]

    #Convert to DataFrame
    df_train_pca = pd.DataFrame(X_train_pca, columns=pca_cols, index=X_train.index)

    if X_test is not None:
        X_test_pca = pca.transform(X_test)
        df_test_pca = pd.DataFrame(X_test_pca, columns=pca_cols, index=X_test.index)
    else:
        df_test_pca = None

    if return_model:
        if df_test_pca is not None:
            return df_train_pca, df_test_pca, pca.explained_variance_ratio_, pca
        return df_train_pca, pca.explained_variance_ratio_, pca

    if df_test_pca is not None:
        return df_train_pca, df_test_pca, pca.explained_variance_ratio_
    return df_train_pca, pca.explained_variance_rati
