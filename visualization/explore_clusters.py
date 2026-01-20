from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    pass
from .style import UNIFORM_BLUE, PALE_PINK

#---Function : plot_cluster_diagnostics ---
def plot_cluster_diagnostics(df_scaled, labels, model_name="Clustering Model"):
    """
    Displays a 1x3 validation dashboard for a clustering model.
    Computes internal validity metrics and visual diagnostics.

    Internal Validity Check for Clustering.
    Evaluates the cohesion and separation of clusters using mathematical
    internal metrics.
    """

    # Imports & style
    sns.set_theme(style="whitegrid")

    # Basic checks
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])

    if n_clusters < 2:
        print("Performance Check Failed: The model produced fewer than 2 clusters.")
        return None

    # Internal validity metrics
    mask = labels != -1

    sil_score = silhouette_score(df_scaled[mask], labels[mask])
    ch_score = calinski_harabasz_score(df_scaled[mask], labels[mask])
    db_index = davies_bouldin_score(df_scaled[mask], labels[mask])

    print("\n--- Clustering Performance Check ---")
    print(f"Model                     : {model_name}")
    print(f"Number of Clusters         : {n_clusters}")
    print(f"Silhouette Score           : {sil_score:.4f}  (Goal: -> 1.0)")
    print(f"Calinski-Harabasz Index    : {ch_score:.2f} (Goal: High)")
    print(f"Davies-Bouldin Index       : {db_index:.4f}  (Goal: -> 0.0)")
    print("-" * 45)

    # Formal diagnostic based on silhouette score
    if sil_score > 0.50:
        print("Status: EXCELLENT. Strong and well-separated cluster structure.")
    elif sil_score > 0.25:
        print("Status: ACCEPTABLE. Moderate structure detected; some overlap likely.")
    elif sil_score > 0:
        print("Status: WEAK. Poorly defined clusters; high risk of overlap.")
    else:
        print("Status: INVALID. Model failed to capture a meaningful structure.")
    print("-" * 45)

    # Visualization dashboard
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    plt.subplots_adjust(wspace=0.3)

    custom_colors = [UNIFORM_BLUE, PALE_PINK, "#9b59b6", "#34495e", "#16a085"]

    # Elbow method or K-distance
    ax0 = axes[0]
    if any(x in model_name.upper() for x in ["DBSCAN", "OPTICS"]):
        from sklearn.neighbors import NearestNeighbors
        k_neighbors = 4
        neigh = NearestNeighbors(n_neighbors=k_neighbors)
        nbrs = neigh.fit(df_scaled)
        distances, _ = nbrs.kneighbors(df_scaled)
        sorted_distances = np.sort(distances[:, k_neighbors - 1], axis=0)
        ax0.plot(sorted_distances, color=custom_colors[0])
        ax0.set_title(f"K-Distance Plot (k={k_neighbors})", fontsize=14)
    else:
        wcss = []
        cluster_range = range(1, 11)
        sample_size = min(len(df_scaled), 5000)
        df_sample = df_scaled.sample(sample_size, random_state=42) if len(df_scaled) > 5000 else df_scaled

        for i in cluster_range:
            kmeans_temp = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans_temp.fit(df_sample)
            wcss.append(kmeans_temp.inertia_)

        ax0.plot(cluster_range, wcss, marker='o', linestyle='--', color=custom_colors[0])
        ax0.axvline(x=n_clusters, color='red', linestyle=':', label=f'Selected: {n_clusters}')
        ax0.set_title("Elbow Method (Inertia)", fontsize=14)
        ax0.legend()

    # Silhouette analysis
    ax1 = axes[1]
    sil_values = silhouette_samples(df_scaled[mask], labels[mask])
    y_lower = 10

    for i, cluster_id in enumerate(np.unique(labels[mask])):
        ith_cluster_sil = sil_values[labels[mask] == cluster_id]
        ith_cluster_sil.sort()
        y_upper = y_lower + ith_cluster_sil.shape[0]
        color = custom_colors[i % len(custom_colors)]
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_sil,
            facecolor=color,
            alpha=0.7
        )
        y_lower = y_upper + 10

    ax1.axvline(x=sil_score, color="red", linestyle="--", label=f"Avg: {sil_score:.2f}")
    ax1.set_title("Silhouette Profile (Cohesion)", fontsize=14)
    ax1.legend()

    # Cluster distribution
    ax2 = axes[2]
    counts = pd.Series(labels).value_counts().sort_index()
    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        ax=ax2,
        palette=custom_colors[:len(counts)],
        hue=counts.index.astype(str),
        legend=False
    )
    ax2.set_title("Distribution per Cluster", fontsize=14)

    plt.suptitle(f"Clustering Diagnostics Dashboard: {model_name}", fontsize=18, y=1.05)
    plt.tight_layout()
    plt.show()

    return {
        "silhouette": sil_score,
        "calinski_harabasz": ch_score,
        "davies_bouldin": db_index,
        "n_clusters": n_clusters
    }

#--- Function : plot_cluster_projections ---
def plot_cluster_projections(df_scaled, labels, model_name="Champion Model"):
    """
    Displays 2D and 3D cluster projections using PCA with unified colors.
    """
    pca_2d = PCA(n_components=2).fit_transform(df_scaled)
    pca_3d = PCA(n_components=3).fit_transform(df_scaled)
    custom_colors = [UNIFORM_BLUE, PALE_PINK, "#9b59b6", "#34495e", "#16a085"]
    fig = plt.figure(figsize=(20, 8))

    #2D Projection
    ax1 = fig.add_subplot(1, 2, 1)
    sns.scatterplot(
        x=pca_2d[:, 0], y=pca_2d[:, 1], 
        hue=labels, palette=custom_colors[:len(np.unique(labels))], 
        ax=ax1, s=60, alpha=0.7, edgecolor='white'
    )
    ax1.set_title("2D PCA Projection", fontsize=14)

    #3D Projection
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #English comment: Map labels to hex colors to ensure 3D consistency
    mapped_colors = [custom_colors[l % len(custom_colors)] for l in labels]
    ax2.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], c=mapped_colors, s=40, alpha=0.6)
    ax2.set_title("3D PCA Projection", fontsize=14)
    plt.suptitle(f"Spatial Separation: {model_name}", fontsize=18, y=1.02)
    plt.show()

#--- Function : plot_cluster_radar_charts ---
def plot_cluster_radar_charts(df_scaled, labels, feature_names):
    """
    Displays Radar Charts representing the DNA of each cluster with unified colors.
    """
    df_temp = df_scaled[feature_names].copy()
    df_temp['Cluster'] = labels
    cluster_means = df_temp.groupby('Cluster').mean()
    unique_labels = np.unique(labels[labels != -1])
    num_vars = len(feature_names)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    custom_colors = [UNIFORM_BLUE, PALE_PINK, "#9b59b6", "#34495e", "#16a085"]
    
    # Determine subplot grid dynamically
    n_clusters = len(unique_labels)
    n_cols = 2
    n_rows = (n_clusters + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), subplot_kw=dict(polar=True))
    
    # Flatten axes and hide extras
    axes_flat = axes.flatten() if n_clusters > 1 else [axes]
    for i in range(len(axes_flat)):
        if i >= n_clusters:
            axes_flat[i].axis('off')  # hide empty subplots
    
    for i, cluster_id in enumerate(unique_labels):
        ax = axes_flat[i]
        values = cluster_means.loc[cluster_id].values.flatten().tolist()
        values += values[:1]
        
        # Apply unified color to plot and fill
        ax.fill(angles, values, color=custom_colors[i % len(custom_colors)], alpha=0.3)
        ax.plot(angles, values, color=custom_colors[i % len(custom_colors)], linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, fontsize=10)
        ax.set_title(f"Cluster {cluster_id}", size=15, color=custom_colors[i % len(custom_colors)], y=1.1)
        ax.set_ylim(df_scaled[feature_names].min().min(), df_scaled[feature_names].max().max())

    plt.suptitle("Behavioral DNA per Cluster", fontsize=20, y=1.02)
    plt.show()
