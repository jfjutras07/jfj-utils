import pandas as pd
import numpy as np
import pytest
from data_preprocessing.diagnostics import premodeling_clustering_check

#--- Test: premodeling_clustering_check ---
def test_premodeling_clustering_check():
    """
    Test the pre-modeling diagnostic report for clustering tasks.
    Focuses on scale differences, outliers, and feature redundancy.
    """
    # 1. Setup synthetic dataset with clustering-specific issues
    # - Scale issue: 'feature_large' has much higher variance than 'feature_small'
    # - Redundancy: 'feat_1' and 'feat_2' are identical
    # - Outliers: 'feature_outlier' has extreme values
    # - Types: 'category_col' is non-numeric
    data = pd.DataFrame({
        "feature_large": np.random.normal(1000, 500, 100),
        "feature_small": np.random.normal(1, 0.1, 100),
        "feat_1": np.linspace(0, 1, 100),
        "feat_2": np.linspace(0, 1, 100),
        "feature_outlier": [10] * 95 + [500, 600, 700, 800, 900],
        "category_col": ["Group_A"] * 100
    })

    # 2. Execute check
    report = premodeling_clustering_check(
        data, 
        corr_threshold=0.9, 
        iqr_multiplier=1.5
    )

    # 3. Assertions on report content
    # Check for the Scale & Magnitude section
    assert "#Scale & Magnitude check" in report
    assert "Warning: Large difference in feature scales detected" in report
    assert "feature_large" in report

    # Check for Feature types (Clustering needs numbers)
    assert "#Feature types" in report
    assert "category_col" in report

    # Check for Outliers (Distort centroids)
    assert "#Outliers" in report
    assert "feature_outlier: 5 points" in report

    # Check for Redundancy (Weights distance twice)
    assert "#Feature Redundancy" in report
    assert "feat_1 ↔ feat_2" in report

    # Check Final Assessment
    assert "#Final assessment" in report
    assert "Clustering is highly sensitive to Scaling" in report

    # 4. Test "Balanced" dataset logic
    balanced_data = pd.DataFrame({
        "f1": np.random.normal(0, 1, 100),
        "f2": np.random.normal(0, 1, 100)
    })
    clean_report = premodeling_clustering_check(balanced_data)
    
    assert "Feature scales appear relatively consistent" in clean_report
    assert "No redundant features detected" in clean_report
