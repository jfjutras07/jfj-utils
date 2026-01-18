import pandas as pd
import numpy as np
import pytest
from data_preprocessing.diagnostics import premodeling_classification_check

#--- Test: premodeling_classification_check ---
def test_premodeling_classification_check():
    """
    Test the pre-modeling diagnostic report for classification.
    Verifies detection of missing values, imbalance, redundancy, and outliers.
    """
    # 1. Setup synthetic dataset with specific issues
    # - Missing values in 'feature_a'
    # - Redundancy between 'feature_b' and 'feature_c' (r=1.0)
    # - Class imbalance (90/10)
    # - Non-numeric column 'city'
    # - Outliers in 'feature_d'
    data = pd.DataFrame({
        "feature_a": [1.0, 2.0, np.nan, 4.0, 5.0] * 20,
        "feature_b": np.linspace(0, 100, 100),
        "feature_c": np.linspace(0, 100, 100),
        "feature_d": [0] * 95 + [1000, -1000, 500, 600, 700],
        "city": ["Paris"] * 100,
        "target": [0] * 90 + [1] * 10
    })

    # 2. Execute check
    report = premodeling_classification_check(
        data, 
        target='target', 
        corr_threshold=0.9, 
        imbalance_threshold=0.15
    )

    # 3. Assertions on report content
    # Check for sections (using the '#' headers defined in the function)
    assert "#Missing values" in report
    assert "feature_a: 20" in report

    assert "#Feature types" in report
    assert "- city" in report

    assert "#Class balance" in report
    assert "Significant class imbalance detected" in report
    assert "Class '1': 10.00%" in report

    assert "#Outliers" in report
    assert "feature_d" in report

    assert "#Feature Redundancy" in report
    assert "feature_b ↔ feature_c" in report

    assert "#Final assessment" in report

    # 4. Test "Clean" dataset logic
    clean_data = pd.DataFrame({
        "f1": np.random.rand(100),
        "f2": np.random.rand(100),
        "target": [0, 1] * 50
    })
    clean_report = premodeling_classification_check(clean_data, target='target')
    
    assert "No major issues detected" in clean_report
    assert "No missing values detected" in clean_report
