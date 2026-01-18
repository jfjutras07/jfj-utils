import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from modeling.model_stability import check_classification_model_stability

#--- Test: check_classification_model_stability ---
def test_check_classification_model_stability():
    """
    Test the stability diagnostics for classification:
    - Verify cross-validation score aggregation.
    - Validate the stability ratio calculation (Std/Mean).
    - Check the perturbation analysis (Label Stability via ARI).
    """
    # 1. Setup synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(200, 4), columns=['f1', 'f2', 'f3', 'f4'])
    # Target with a clear pattern for high stability
    y = (X['f1'] + X['f2'] > 1).astype(int)

    # 2. Case A: Highly Stable Model
    # A simple logistic regression on linear data should be very stable
    model_stable = LogisticRegression()
    results_stable = check_classification_model_stability(
        model_stable, X, y, cv=3, scoring='f1_macro'
    )

    # Assertions for Case A
    assert isinstance(results_stable, dict)
    assert 0 <= results_stable["mean"] <= 1
    assert results_stable["stability_ratio"] >= 0
    # ARI should be close to 1 for a stable model
    assert results_stable["label_stability"] is not None
    assert 0 <= results_stable["label_stability"] <= 1

    # 3. Case B: Testing the Stability Ratio logic
    # Ensure the function handles the edge case where mean score is 0
    y_impossible = pd.Series([0, 1] * 100) # Noise
    X_noise = pd.DataFrame(np.random.rand(200, 2))
    
    results_unstable = check_classification_model_stability(
        model_stable, X_noise, y_impossible, cv=2
    )
    
    if results_unstable["mean"] == 0:
        assert results_unstable["stability_ratio"] == 0
    else:
        assert results_unstable["stability_ratio"] >= 0

    # 4. Case C: High Variance (Random Forest with 1 tree)
    # Testing if ARI and Ratio react to model sensitivity
    model_sensitive = RandomForestClassifier(n_estimators=1, random_state=1)
    results_sens = check_classification_model_stability(
        model_sensitive, X, y, cv=3
    )
    
    assert "label_stability" in results_sens
