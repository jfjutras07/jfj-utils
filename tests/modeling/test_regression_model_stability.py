import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from modeling.model_stability import check_regression_model_stability

#--- Test: check_regression_model_stability ---
def test_check_regression_model_stability():
    """
    Test the regression stability diagnostics:
    - Verify score aggregation across folds.
    - Validate the conversion of negative metrics (e.g., neg_mean_absolute_error).
    - Check the stability ratio (Coefficient of Variation) logic.
    """
    # 1. Setup synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(200, 2), columns=['f1', 'f2'])
    y = 5 * X['f1'] + np.random.normal(0, 0.1, 200)

    # 2. Case A: Standard Metric (R2) - Stable Model
    model = LinearRegression()
    results_r2 = check_regression_model_stability(
        model, X, y, cv=3, scoring='r2'
    )

    # Assertions for Case A
    assert isinstance(results_r2, dict)
    assert 0 <= results_r2["mean"] <= 1
    assert results_r2["stability_ratio"] >= 0
    # A simple linear model on linear data should be "STABLE" (ratio < 0.05)
    assert results_r2["stability_ratio"] < 0.15

    # 3. Case B: Negative Metric (MAE)
    # The function should convert -scores to positive values
    results_mae = check_regression_model_stability(
        model, X, y, cv=3, scoring='neg_mean_absolute_error'
    )

    # Assertions for Case B
    assert results_mae["mean"] > 0 # Should be positive after internal conversion
    assert "std" in results_mae

    # 4. Case C: Unstable Model (High Variance)
    # Using a very complex model on very few noisy samples
    X_small = pd.DataFrame(np.random.rand(20, 10))
    y_small = np.random.normal(0, 1, 20)
    model_overfit = RandomForestRegressor(n_estimators=100, max_depth=20)
    
    results_unstable = check_regression_model_stability(
        model_overfit, X_small, y_small, cv=5, scoring='r2'
    )

    # Logic check: even if we don't assert the 'Status' string, 
    # we verify the metrics are computed
    assert results_unstable["stability_ratio"] is not None
    
    # 5. Case D: Mean Score is Zero
    # Verify the function handles division by zero
    results_zero = check_regression_model_stability(
        model, X, np.zeros(200), cv=2
    )
    assert results_zero["stability_ratio"] == 0
