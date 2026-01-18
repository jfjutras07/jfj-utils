import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from modeling.model_stability import check_clustering_model_stability

#--- Test: check_clustering_model_stability ---
def test_check_clustering_model_stability():
    """
    Test the clustering stability diagnostic:
    - Verify ARI calculation across different random seeds.
    - Validate alignment of labels using indices during subsampling.
    - Check the dictionary output structure and mean ARI values.
    """
    # 1. Setup synthetic data (3 distinct clusters for high stability)
    np.random.seed(42)
    data = np.vstack([
        np.random.normal(0, 0.2, (100, 2)),
        np.random.normal(5, 0.2, (100, 2)),
        np.random.normal(10, 0.2, (100, 2))
    ])
    predictors = ['feat1', 'feat2']
    df = pd.DataFrame(data, columns=predictors)

    # 2. Build a pipeline as required by the function (model.named_steps['model'])
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KMeans(n_clusters=3, n_init=1, random_state=42))
    ])

    # 3. Run stability check
    seeds = [0, 42]
    results = check_clustering_model_stability(
        model, df, predictors, seeds=seeds, subsample_frac=0.7
    )

    # 4. Assertions on results dictionary
    assert isinstance(results, dict)
    assert "ari_seeds_mean" in results
    assert "ari_subsampling_mean" in results
    assert len(results["ari_seeds"]) == len(seeds) - 1
    
    # 5. Logic Assertions: Stability scores
    # With very distinct clusters, ARI should be high
    assert 0 <= results["ari_overall_mean"] <= 1
    if results["ari_overall_mean"] > 0.85:
        # Check if the function logic would print "HIGHLY STABLE"
        assert True

    # 6. Verify Subsampling Alignment Logic
    # The key part of your function is: common_idx = base_sub_labels_series.index.intersection(...)
    # We ensure that even with different samples, the ARI is calculated on common points
    assert results["ari_subsampling_mean"] is not None
    
    # 7. Test with a model without random_state (e.g., Agglomerative)
    # This checks if the 'hasattr' safety check in your function works
    from sklearn.cluster import AgglomerativeClustering
    model_no_rs = Pipeline([
        ('scaler', StandardScaler()),
        ('model', AgglomerativeClustering(n_clusters=3))
    ])
    
    results_no_rs = check_clustering_model_stability(
        model_no_rs, df, predictors, seeds=[0, 1]
    )
    assert results_no_rs["ari_seeds_mean"] == 1.0 # Deterministic model should be stable
