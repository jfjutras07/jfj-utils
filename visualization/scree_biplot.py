import matplotlib.pyplot as plt
import numpy as np
from .style import UNIFORM_BLUE, PALE_PINK

#--- Function : plot_scree_biplot ---
def plot_scree_biplot(pca, df_pca, feature_names, scale=3, figsize=(16, 6)):
    """
    Visualization of PCA results: Scree plot for variance and Biplot for projections.
    """
    pca_components = pca.components_
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Scree Plot ---
    x_range = range(1, len(explained_var) + 1)
    axes[0].plot(x_range, explained_var, 'o-', linewidth=2, color=UNIFORM_BLUE, label='Individual')
    axes[0].plot(x_range, cumulative_var, 's--', linewidth=2, color=PALE_PINK, label='Cumulative')
    
    axes[0].set_title('Scree Plot - Explained Variance', fontsize=14)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Variance Ratio')
    axes[0].set_xticks(x_range)
    axes[0].legend(loc='best')

    # --- PCA Biplot (PC1 vs PC2) ---
    # Observation points
    axes[1].scatter(df_pca.iloc[:, 0], df_pca.iloc[:, 1], alpha=0.5, color=UNIFORM_BLUE, edgecolor='white')
    
    # Feature vectors (loading arrows)
    for i, var in enumerate(feature_names):
        dx = pca_components[0, i] * scale
        dy = pca_components[1, i] * scale
        
        axes[1].arrow(0, 0, dx, dy, color=PALE_PINK, alpha=0.8, head_width=0.08, linewidth=1.5)
        axes[1].text(dx * 1.15, dy * 1.15, var, color='black', fontsize=10, 
                     fontweight='bold', ha='center', va='center')

    axes[1].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
    axes[1].set_title('PCA Biplot: Projections & Loadings', fontsize=14)
    
    # Origin lines for better orientation
    axes[1].axhline(0, color='grey', linestyle='--', alpha=0.3)
    axes[1].axvline(0, color='grey', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()
