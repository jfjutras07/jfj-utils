import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from visualization.explore_clusters import plot_cluster_diagnostics

#--- Test: plot_cluster_diagnostics ---
def test_plot_cluster_diagnostics():
    """
    Test the clustering diagnostic dashboard:
    - Verify calculation of Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics.
    - Check handling of valid/invalid cluster counts.
    - Validate the output dictionary structure.
    """
    # 1. Setup synthetic clustering data (3 clear blobs)
    np.random.seed(42)
    data = np.vstack([
        np.random.normal(0, 0.1, (50, 2)),
        np.random.normal(3, 0.1, (50, 2)),
        np.random.normal(6, 0.1, (50, 2))
    ])
    df_scaled = pd.DataFrame(data, columns=['feat1', 'feat2'])
    
    # 2. Case A: Standard Clustering (KMeans with k=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    
    plt.ion() # Prevent plot from blocking
    results = plot_cluster_diagnostics(df_scaled, labels, model_name="KMeans_Test")
    plt.close('all')

    # Assertions for Case A
    assert isinstance(results, dict)
    assert results["n_clusters"] == 3
    assert 0 <= results["silhouette"] <= 1
    assert results["calinski_harabasz"] > 0
    assert results["davies_bouldin"] >= 0

    # 3. Case B: Invalid Clustering (Only 1 cluster)
    labels_single = np.zeros(150)
    results_single = plot_cluster_diagnostics(df_scaled, labels_single)
    
    # Should return None and print a failure message
    assert results_single is None

    # 4. Case C: Handling Noise (Labels with -1, like DBSCAN)
    # Adding some noise points (-1)
    labels_with_noise = labels.copy()
    labels_with_noise[:5] = -1
    
    results_noise = plot_cluster_diagnostics(df_scaled, labels_with_noise, model_name="DBSCAN_Test")
    plt.close('all')

    # Assertions for Case C
    # Metrics should be calculated excluding the noise points (-1)
    assert results_noise["n_clusters"] == 3
    assert "silhouette" in results_noise
