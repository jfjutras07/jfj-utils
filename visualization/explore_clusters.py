from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from .style import UNIFORM_BLUE, PALE_PINK

def plot_cluster_diagnostics(df_scaled, labels, model_name="Champion Model"):
    """
    Displays a 1x3 validation dashboard for a single clustering model.
    Left: Elbow Method or K-Distance | Middle: Silhouette Analysis | Right: Cluster Distribution
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    plt.subplots_adjust(wspace=0.3)
    
    unique_labels = np.unique(labels)
    num_clusters_selected = len(unique_labels[unique_labels != -1])
    # English comment: Define custom colors for visualization consistency
    custom_colors = [UNIFORM_BLUE, PALE_PINK, "#9b59b6", "#34495e", "#16a085"]

    # --- Left: Elbow Method or K-Distance ---
    ax0 = axes[0]
    if any(x in model_name.upper() for x in ["DBSCAN", "OPTICS"]):
        # English comment: For density-based models, use K-distance to validate Epsilon
        from sklearn.neighbors import NearestNeighbors
        k_neighbors = 4
        neigh = NearestNeighbors(n_neighbors=k_neighbors)
        nbrs = neigh.fit(df_scaled)
        distances, _ = nbrs.kneighbors(df_scaled)
        sorted_distances = np.sort(distances[:, k_neighbors-1], axis=0)
        
        ax0.plot(sorted_distances, color=UNIFORM_BLUE)
        ax0.set_title(f"K-Distance Plot (k={k_neighbors})", fontsize=14)
        ax0.set_xlabel("Points sorted by distance")
        ax0.set_ylabel("Epsilon Distance")
    else:
        # English comment: Standard Elbow method for centroid-based algorithms
        wcss = []
        cluster_range = range(1, 11)
        sample_size = min(len(df_scaled), 50000)
        df_sample = df_scaled.sample(sample_size, random_state=42) if len(df_scaled) > 50000 else df_scaled
        
        for i in cluster_range:
            kmeans_temp = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            kmeans_temp.fit(df_sample)
            wcss.append(kmeans_temp.inertia_)
        
        ax0.plot(cluster_range, wcss, marker='o', linestyle='--', color=UNIFORM_BLUE)
        if num_clusters_selected in cluster_range:
            ax0.axvline(x=num_clusters_selected, color='red', linestyle=':', label=f'Selected: {num_clusters_selected}')
        ax0.set_title("Elbow Method (Inertia)", fontsize=14)
        ax0.set_xlabel("Number of Clusters")
        ax0.set_ylabel("WCSS")
        ax0.legend()

    # --- Middle: Silhouette Analysis ---
    ax1 = axes[1]
    # English comment: Filter noise (-1) to avoid skewed silhouette visualization
    mask = labels != -1
    if len(np.unique(labels[mask])) > 1:
        sil_values = silhouette_samples(df_scaled[mask], labels[mask])
        sil_avg = silhouette_score(df_scaled[mask], labels[mask])
        y_lower = 10
        
        for i, cluster_id in enumerate(np.unique(labels[mask])):
            ith_cluster_sil = sil_values[labels[mask] == cluster_id]
            ith_cluster_sil.sort()
            y_upper = y_lower + ith_cluster_sil.shape[0]
            color = custom_colors[i % len(custom_colors)]
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil, facecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * ith_cluster_sil.shape[0], str(cluster_id))
            y_lower = y_upper + 10
            
        ax1.axvline(x=sil_avg, color="red", linestyle="--", label=f'Avg Score: {sil_avg:.2f}')
    ax1.set_title("Silhouette Profile (Cohesion)", fontsize=14)
    ax1.set_xlabel("Silhouette coefficient")
    ax1.set_ylabel("Cluster ID")
    ax1.legend()

    # --- Right: Data Distribution ---
    ax2 = axes[2]
    counts = pd.Series(labels).value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax2, palette=custom_colors, hue=counts.index.astype(str), legend=False)
    ax2.set_title("Distribution per Cluster (Population)", fontsize=14)
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Number of Observations")

    plt.suptitle(f"Clustering Diagnostics Dashboard: {model_name}", fontsize=18, y=1.05)
    plt.tight_layout()
    plt.show()

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

#--- Function : plot_cluster_radar_charts ---
def plot_cluster_radar_charts(df_scaled, labels, feature_names):
    """
    Displays a 2x2 grid of Radar Charts representing the DNA of each cluster.
    """
    # Calculate the mean of features for each cluster
    df_temp = df_scaled[feature_names].copy()
    df_temp['Cluster'] = labels
    cluster_means = df_temp.groupby('Cluster').mean()
    
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    num_vars = len(feature_names)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1] # Close the loop

    # Colors from your style
    try:
        from .style import UNIFORM_BLUE, PALE_PINK
        custom_colors = [UNIFORM_BLUE, PALE_PINK, "#9b59b6", "#34495e"]
    except:
        custom_colors = ["#3498db", "#e74c3c", "#9b59b6", "#34495e"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw=dict(polar=True))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    axes_flat = axes.flatten()

    for i, cluster_id in enumerate(unique_labels):
        if i >= 4: break # Limit to 2x2 grid
        
        ax = axes_flat[i]
        values = cluster_means.loc[cluster_id].values.flatten().tolist()
        values += values[:1] # Close the loop
        
        # Draw the chart
        ax.fill(angles, values, color=custom_colors[i % len(custom_colors)], alpha=0.3)
        ax.plot(angles, values, color=custom_colors[i % len(custom_colors)], linewidth=2)
        
        # Fix labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, fontsize=10)
        
        # Title and style
        ax.set_title(f"Cluster {cluster_id} Profile", size=15, color=custom_colors[i % len(custom_colors)], y=1.1)
        ax.set_ylim(df_scaled[feature_names].min().min(), df_scaled[feature_names].max().max())

    plt.suptitle("Behavioral DNA per Cluster", fontsize=20, y=1.02)
    plt.show()
