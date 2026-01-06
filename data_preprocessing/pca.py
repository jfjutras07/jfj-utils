from sklearn.decomposition import PCA
import pandas as pd

#--- Function : perform_pca (adapted) ---
def perform_pca(X_scaled, variance_threshold=0.95, return_model=False):
    """
    Perform PCA on a pre-scaled dataset and return a DataFrame with principal components.
    
    Parameters:
    -----------
    X_scaled : np.array or pd.DataFrame
        Scaled numeric data (standardized)
    variance_threshold : float, default=0.95
        Fraction of total variance to retain (determines number of components)
    return_model : bool, default=False
        Whether to return the fitted PCA model (needed to transform test sets)
    
    Returns:
    --------
    df_pca : pd.DataFrame
        DataFrame containing principal components
    explained_variance_ratio : np.array
        Variance explained by each principal component
    pca_model : sklearn.decomposition.PCA (optional)
        The fitted PCA model (returned only if return_model=True)
    """
    
    #Initialize PCA
    pca = PCA(n_components=variance_threshold)
    
    #Fit PCA on the scaled data
    X_pca = pca.fit_transform(X_scaled)
    
    #Create column names for principal components
    pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    
    #Convert to DataFrame
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    
    if return_model:
        return df_pca, pca.explained_variance_ratio_, pca
    else:
        return df_pca, pca.explained_variance_ratio_
