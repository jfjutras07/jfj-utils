import pandas as pd
import numpy as np
import io
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from eda.explainability import interaction_effects

#--- Function : test_interaction_effects ---
def test_interaction_effects():
    # Sample DataFrame
    df = pd.DataFrame({
        "y": np.random.rand(20),
        "x1": np.random.rand(20),
        "x2": np.random.rand(20),
        "x3": np.random.rand(20)
    })

    predictors = ["x1", "x2", "x3"]
    X_train = df[predictors]
    y_train = df["y"]

    # Train a simple model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Run interaction_effects
    interaction_effects(model, df, predictors, top_n=2)

    # Reset stdout
    sys.stdout = sys.__stdout__

    output = captured_output.getvalue()
    assert "--- Top Interaction Analysis for:" in output
    for feat in ["x1", "x2", "x3"]:
        assert feat in output or len(output.split(":")[-1].split(",")) <= 2  # top_n respected
