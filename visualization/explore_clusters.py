from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from .style import UNIFORM_BLUE, PALE_PINK

#--- Function : plot_cluster_projections ---
def plot_cluster_projections(df_scaled, labels, model_name="Champion Model"):
    """
    Displays 2D and 3D cluster projections side-by-side using PCA.
    Helps visualize spatial separation and overlap between groups.
    """
    print(f"Generating spatial projections for: {model_name}...")
    print("-" * 50)

    # PCA Projections
    pca_2d = PCA(n_components=2).fit_transform(df_scaled)
    pca_3d = PCA(n_components=3).fit_transform(df_scaled)
    
    # Style setup
    custom_colors = [UNIFORM_BLUE, PALE_PINK, "#9b59b6", "#34495e", "#16a085"]
    fig = plt.figure(figsize=(20, 8))

    # 2D Projection
    ax1 = fig.add_subplot(1, 2, 1)
    sns.scatterplot(
        x=pca_2d[:, 0], y=pca_2d[:, 1], 
        hue=labels, palette=custom_colors[:len(np.unique(labels))], 
        ax=ax1, s=60, alpha=0.7, edgecolor='white'
    )
    ax1.set_title("2D PCA Projection", fontsize=14)
    ax1.set_xlabel("First Principal Component")
    ax1.set_ylabel("Second Principal Component")
    ax1.legend(title="Cluster", loc='best')

    # 3D Projection
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    scatter = ax2.scatter(
        pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], 
        c=[custom_colors[l % len(custom_colors)] for l in labels],
        s=40, alpha=0.6, edgecolor='white'
    )
    ax2.set_title("3D PCA Projection", fontsize=14)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")

    plt.suptitle(f"Spatial Cluster Separation: {model_name}", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

#--- Function : plot_cluster_diagnostics ---
def plot_cluster_diagnostics(df_scaled, labels, model_name="Champion Model"):
    """
    Displays a 1x2 validation dashboard for a single clustering model.
    Left: Silhouette Analysis (Cohesion) | Right: Employee Distribution (Population)
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.3)
    
    # Silhouette Analysis
    ax1 = axes[0]
    sil_values = silhouette_samples(df_scaled, labels)
    sil_avg = silhouette_score(df_scaled, labels)
    y_lower = 10
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    
    custom_colors = [UNIFORM_BLUE, PALE_PINK, "#9b59b6", "#34495e", "#16a085"]
    
    for i, cluster_id in enumerate(unique_labels):
        if cluster_id == -1: continue 
        
        ith_cluster_sil = sil_values[labels == cluster_id]
        ith_cluster_sil.sort()
        y_upper = y_lower + ith_cluster_sil.shape[0]
        color = custom_colors[i % len(custom_colors)]
        
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil, facecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * ith_cluster_sil.shape[0], str(cluster_id))
        y_lower = y_upper + 10
        
    ax1.axvline(x=sil_avg, color="red", linestyle="--", label=f'Average: {sil_avg:.2f}')
    ax1.set_title("Silhouette Profile (Cohesion)", fontsize=14)
    ax1.set_xlabel("Silhouette coefficient")
    ax1.set_ylabel("Cluster")
    ax1.legend()

    # Employee Distribution
    ax2 = axes[1]
    counts = pd.Series(labels).value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, ax=ax2, palette=custom_colors, hue=counts.index, legend=False)
    ax2.set_title("Employee Distribution per Cluster", fontsize=14)
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Count")

    plt.suptitle(f"Final Clustering Diagnostics: {model_name}", fontsize=18, y=1.05)
    plt.tight_layout()
    plt.show()
