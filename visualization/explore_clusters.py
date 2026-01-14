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

def plot_cluster_diagnostics(df_scaled, labels=None, model_name="KMeans", tune_grid=None):
    """
    Displays a 1x3 validation dashboard for a single clustering model.
    Optimizes the model if tune_grid is provided for: KMeans, Agglomerative, GMM, BIRCH, DBSCAN, K-Medoids.
    """
    #Optimization Logic
    #English comment: If tune_grid is provided, search for the best silhouette score
    if tune_grid is not None:
        best_score = -1
        best_labels = None
        best_params = None
        
        #Standardize iteration logic for K-based models vs Density-based
        if any(m in model_name.upper() for m in ["KMEANS", "AGGLOMERATIVE", "GMM", "BIRCH", "KMEDOIDS"]):
            for k in tune_grid.get('k', [2, 3, 4, 5]):
                if "KMEANS" in model_name.upper():
                    m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(df_scaled)
                    l = m.labels_
                elif "AGGLOMERATIVE" in model_name.upper():
                    l = AgglomerativeClustering(n_clusters=k).fit_predict(df_scaled)
                elif "GMM" in model_name.upper():
                    l = GaussianMixture(n_components=k, random_state=42).fit_predict(df_scaled)
                elif "BIRCH" in model_name.upper():
                    l = Birch(n_clusters=k).fit_predict(df_scaled)
                elif "KMEDOIDS" in model_name.upper():
                    l = KMedoids(n_clusters=k, random_state=42).fit_predict(df_scaled)
                
                score = silhouette_score(df_scaled, l)
                if score > best_score:
                    best_score, best_labels, best_params = score, l, k
            model_name = f"{model_name} (Best k={best_params})"
            
        elif "DBSCAN" in model_name.upper():
            for eps in tune_grid.get('eps', [0.3, 0.5]):
                for ms in tune_grid.get('min_samples', [10, 50]):
                    l = DBSCAN(eps=eps, min_samples=ms).fit_predict(df_scaled)
                    n_c = len(set(l)) - (1 if -1 in l else 0)
                    if n_c > 1:
                        mask = l != -1
                        score = silhouette_score(df_scaled[mask], l[mask])
                        if score > best_score:
                            best_score, best_labels, best_params = score, l, (eps, ms)
            model_name = f"{model_name} (Best eps={best_params[0]}, ms={best_params[1]})"
        
        labels = best_labels

    #Standard Visualization Logic
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    plt.subplots_adjust(wspace=0.3)
    
    unique_labels = np.unique(labels)
    num_clusters_selected = len(unique_labels[unique_labels != -1])
    #Custom colors (Replace UNIFORM_BLUE/PALE_PINK with hex if not defined)
    custom_colors = ["#4682B4", "#FFD1DC", "#9b59b6", "#34495e", "#16a085"]

    #Elbow Method or K-Distance
    ax0 = axes[0]
    if any(x in model_name.upper() for x in ["DBSCAN", "OPTICS"]):
        from sklearn.neighbors import NearestNeighbors
        k_neighbors = 4
        neigh = NearestNeighbors(n_neighbors=k_neighbors)
        nbrs = neigh.fit(df_scaled)
        distances, _ = nbrs.kneighbors(df_scaled)
        sorted_distances = np.sort(distances[:, k_neighbors-1], axis=0)
        ax0.plot(sorted_distances, color=custom_colors[0])
        ax0.set_title(f"K-Distance Plot (k={k_neighbors})", fontsize=14)
    else:
        wcss = []
        cluster_range = range(1, 11)
        sample_size = min(len(df_scaled), 5000)
        df_sample = df_scaled.sample(sample_size, random_state=42) if len(df_scaled) > 5000 else df_scaled
        for i in cluster_range:
            kmeans_temp = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            kmeans_temp.fit(df_sample)
            wcss.append(kmeans_temp.inertia_)
        ax0.plot(cluster_range, wcss, marker='o', linestyle='--', color=custom_colors[0])
        if num_clusters_selected in cluster_range:
            ax0.axvline(x=num_clusters_selected, color='red', linestyle=':', label=f'Selected: {num_clusters_selected}')
        ax0.set_title("Elbow Method (Inertia)", fontsize=14)
        ax0.legend()

    #Middle: Silhouette Analysis
    ax1 = axes[1]
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
            y_lower = y_upper + 10
        ax1.axvline(x=sil_avg, color="red", linestyle="--", label=f'Avg Score: {sil_avg:.2f}')
    ax1.set_title("Silhouette Profile (Cohesion)", fontsize=14)
    ax1.legend()

    #Data Distribution
    ax2 = axes[2]
    counts = pd.Series(labels).value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax2, palette=custom_colors, hue=counts.index.astype(str), legend=False)
    ax2.set_title("Distribution per Cluster", fontsize=14)

    plt.suptitle(f"Clustering Diagnostics Dashboard: {model_name}", fontsize=18, y=1.05)
    plt.tight_layout()
    plt.show()
    
    return labels #Returns the labels of the best optimized model

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
