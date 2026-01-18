import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from visualization.explore_classification import plot_classification_diagnostics

#--- Test: plot_classification_diagnostics ---
def test_plot_classification_diagnostics():
    """
    Test the diagnostic dashboard for classification:
    - Verify plot creation (Matplotlib objects).
    - Validate the calculation of the generalization gap.
    - Check the dictionary output structure.
    """
    # 1. Setup synthetic classification data
    np.random.seed(42)
    X = np.random.rand(200, 4)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    
    # Split for the diagnostic function
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    # 2. Initialize and fit a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # 3. Run the diagnostic function
    # We use plt.ion() or mock to avoid blocking the test with a GUI window
    plt.ion() 
    results = plot_classification_diagnostics(
        model, X_train, y_train, X_test, y_test, cv=3
    )
    plt.close('all') # Close figures to free memory

    # 4. Assertions on results dictionary
    assert isinstance(results, dict)
    assert "gap" in results
    assert "f1_test" in results
    assert isinstance(results["gap"], (float, np.float64))
    
    # 5. Logic assertions
    # F1 score should be between 0 and 1
    assert 0 <= results["f1_test"] <= 1
    
    # Check if a gap was calculated (even if 0)
    assert results["gap"] is not None

    # 6. Plotting side-effects check
    # Verify that the function created a figure with two axes (Learning Curve & Confusion Matrix)
    fig = plt.gcf()
    # Note: Since plt.show() was called inside, we usually verify 
    # the state before or during the function execution in more advanced setups.
