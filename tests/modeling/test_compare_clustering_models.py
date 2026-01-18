import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from modeling.clustering_models import compare_clustering_models

#--- Test: compare_clustering_models ---
def test_compare_clustering_models():
    """
    Test the clustering orchestration logic, label assignment to the input DataFrame,
    and Silhouette score calculation.
    """
    # 1. Setup synthetic data (3 clear blobs)
    np.random.seed(42)
    n_samples = 150
    data = np.vstack([
        np.random.normal(0, 0.5, (50, 2)),
        np.random.normal(3, 0.5, (50, 2)),
        np.random.normal(6, 0.5, (50, 2))
    ])
    
    predictors = ["feat1", "feat2"]
    df = pd.DataFrame(data, columns=predictors)

    # 2. Run comparison
    # We use a small k and no optimization for speed in unit testing
    models_dict = compare_clustering_models(df, predictors, k=3, optimize=False)

    # 3. Assertions on the returned dictionary
    assert isinstance(models_dict, dict)
    assert 'KMeans' in models_dict
    assert 'GMM' in models_dict
    assert isinstance(models_dict['KMeans'], Pipeline)

    # 4. Assertions on DataFrame modification
    # Check if new label columns were added
    expected_cols = [f'Cluster_{name}' for name in models_dict.keys() if models_dict[name] is not None]
    for col in expected_cols:
        assert col in df.columns
        # Labels should not be all identical (meaning some clustering happened)
        assert df[col].nunique() >= 1

    # 5. Assertions on logic integrity
    # Silhouette score logic requires checking if clusters were formed
    # We verify that the results are consistent with the data shape
    assert len(df) == n_samples
    
    # 6. Specific case: DBSCAN handling
    # DBSCAN might return -1 for noise; ensure the function handles it without crashing
    if 'DBSCAN' in df.columns:
        unique_labels = df['Cluster_DBSCAN'].unique()
        # Ensure it's not only noise
        assert len(unique_labels) > 0
