from sklearn.decomposition import PCA
import pandas as pd

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
    return df_train_pca, pca.explained_variance_ratio_
