import matplotlib.pyplot as plt
import numpy as np

#--- Function : scree_biplot ---
def scree_biplot(pca, df_pca, feature_names, scale=3, figsize=(16,6)):
    """
    Plot Scree plot and PCA biplot side by side.
    
    Parameters:
    -----------
    pca : fitted sklearn PCA object
        The PCA object after fitting your data.
    df_pca : pd.DataFrame
        DataFrame containing the PCA-transformed data.
    feature_names : list of str
        List of original feature names corresponding to PCA.
    scale : float
        Scaling factor for the arrows in the biplot.
    figsize : tuple
        Figure size.
    """
    pca_components = pca.components_
    explained_var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    #Scree plot
    axes[0].plot(range(1, len(explained_var)+1), explained_var, 'o-', linewidth=2, color='blue')
    axes[0].plot(range(1, len(explained_var)+1), explained_var.cumsum(), 's--', color='red', label='Cumulative')
    axes[0].set_title('Scree Plot - PCA', fontsize=14)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_xticks(range(1, len(explained_var)+1))
    axes[0].grid(True)
    axes[0].legend()

    #PCA Biplot (PC1 vs PC2)
    axes[1].scatter(df_pca.iloc[:,0], df_pca.iloc[:,1], alpha=0.6)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('PCA Biplot', fontsize=14)
    
    #Add arrows for variable loadings
    for i, var in enumerate(feature_names):
        axes[1].arrow(0, 0, 
                      pca_components[0, i]*scale,
                      pca_components[1, i]*scale,
                      color='red', alpha=0.7, head_width=0.1)
        axes[1].text(pca_components[0, i]*scale*1.1, pca_components[1, i]*scale*1.1, var,
                     color='red', fontsize=12)
    
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()
