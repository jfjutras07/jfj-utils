from sklearn.decomposition import PCA
import pandas as pd

#--- Function : perform_pca ---
def perform_pca(X_scaled, variance_threshold=0.95):
    """
    Perform PCA on a pre-scaled dataset and return a DataFrame with principal components.
    
    Parameters:
    -----------
    X_scaled : np.array or pd.DataFrame
        Scaled numeric data (standardized)
    variance_threshold : float, default=0.95
        Fraction of total variance to retain (determines number of components)
    
    Returns:
    --------
    df_pca : pd.DataFrame
        DataFrame containing principal components
    explained_variance_ratio : np.array
        Variance explained by each principal component
    """
    
    #Initialize PCA
    pca = PCA(n_components=variance_threshold)
    
    #Fit PCA on the scaled data
    X_pca = pca.fit_transform(X_scaled)
    
    #Create column names for principal components
    pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    
    #Convert to DataFrame
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    
    return df_pca, pca.explained_variance_ratio_
