import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from .style import UNIFORM_BLUE, PALE_PINK

#--- Function : plot_cluster_diagnostics ---
def plot_cluster_diagnostics(df_scaled, labels, model_name="Champion Model", inertia_list=None, k_range=None):
    """
    Displays a 2x2 validation dashboard for a single clustering model.
    Top: Silhouette Analysis | Elbow Method (Inertia)
    Bottom: Calinski-Harabasz Index | Davies-Bouldin Index
    """
    print(f"Generating validation dashboard for: {model_name}...")
    print(f"Data points: {df_scaled.shape[0]} | Features: {df_scaled.shape[1]}")
    print("-" * 50)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Silhouette Analysis
    ax1 = axes[0, 0]
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

    # Elbow Method
    ax2 = axes[0, 1]
    if inertia_list and k_range:
        sns.lineplot(x=k_range, y=inertia_list, marker='o', ax=ax2, color=UNIFORM_BLUE)
        ax2.set_title("Elbow Method (Inertia Optimization)", fontsize=14)
        ax2.set_xlabel("Number of clusters (k)")
        ax2.set_ylabel("Inertia")
    else:
        counts = pd.Series(labels).value_counts().sort_index()
        sns.barplot(x=counts.index, y=counts.values, ax=ax2, palette=custom_colors, hue=counts.index, legend=False)
        ax2.set_title("Employee Distribution per Cluster", fontsize=14)
        ax2.set_xlabel("Cluster ID")
        ax2.set_ylabel("Count")

    # Calinski-Harabasz Index
    ax3 = axes[1, 0]
    ch_score = calinski_harabasz_score(df_scaled, labels)
    sns.barplot(x=["Calinski-Harabasz"], y=[ch_score], ax=ax3, color=UNIFORM_BLUE, width=0.4)
    ax3.set_title(f"CH Index: {ch_score:.2f} (Higher is better)", fontsize=12)
    ax3.set_ylabel("Score")

    # Davies-Bouldin Index
    ax4 = axes[1, 1]
    db_score = davies_bouldin_score(df_scaled, labels)
    sns.barplot(x=["Davies-Bouldin"], y=[db_score], ax=ax4, color=PALE_PINK, width=0.4)
    ax4.set_title(f"DB Index: {db_score:.4f} (Lower is better)", fontsize=12)
    ax4.set_ylabel("Score")

    plt.suptitle(f"Final Clustering Diagnostics: {model_name}", fontsize=18, y=0.96)
    plt.show()
    print("Dashboard successfully generated.")
