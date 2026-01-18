import pandas as pd
import numpy as np
import pytest
from data_preprocessing.diagnostics import premodeling_regression_check

#--- Test: premodeling_regression_check ---
def test_premodeling_regression_check():
    """
    Test the pre-modeling diagnostic report for regression.
    Verifies detection of constant columns, PCA outliers, target issues, 
    and multicollinearity.
    """
    # 1. Setup synthetic dataset with regression-specific issues
    # - Constant column: 'const_feat' (should be flagged)
    # - High correlation: 'feat_1' and 'feat_2' (r=1.0)
    # - PCA outliers: 'component_PC1' with extreme values
    # - Target issue: Missing values in 'target'
    # - Size issue: Very small dataframe (n=20)
    data = pd.DataFrame({
        "feat_1": np.linspace(0, 10, 20),
        "feat_2": np.linspace(0, 10, 20),
        "const_feat": [42] * 20,
        "component_PC1": [0] * 18 + [500, -500],
        "non_numeric": ["A"] * 20,
        "target": [1.0, 2.0, np.nan] + [4.0] * 17
    })

    # 2. Execute check
    report = premodeling_regression_check(
        data, 
        target='target', 
        corr_threshold=0.9,
        min_unique_for_continuous=5
    )

    # 3. Assertions on report content
    # Check for constant columns (Feature variance)
    assert "#Feature variance" in report
    assert "const_feat" in report

    # Check for Feature types
    assert "#Feature types" in report
    assert "non_numeric" in report

    # Check for PCA-specific outliers
    assert "##PCA/MCA-derived features" in report
    assert "component_PC1" in report

    # Check for Correlations
    assert "#High correlations" in report
    assert "feat_1 ↔ feat_2" in report

    # Check for Target validation
    assert "#Target validation" in report
    assert "Target contains missing values" in report

    # Check for Dataset size
    assert "#Dataset size" in report
    assert "Dataset may be too small" in report

    # 4. Test "Perfect" dataset logic
    perfect_data = pd.DataFrame({
        "f1": np.random.rand(100),
        "f2": np.random.rand(100),
        "target": np.random.rand(100)
    })
    clean_report = premodeling_regression_check(perfect_data, target='target')
    
    assert "No structural issues detected" in clean_report
    assert "No constant columns detected" in clean_report
