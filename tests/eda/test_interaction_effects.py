import pandas as pd
import numpy as np
import io
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from eda.explainability import interaction_effects

#--- Function : test_interaction_effects ---
def test_interaction_effects():
    # Create synthetic dataset
    df = pd.DataFrame({
        "y": np.random.rand(30),
        "x1": np.random.rand(30),
        "x2": np.random.rand(30),
        "x3": np.random.rand(30)
    })

    predictors = ["x1", "x2", "x3"]

    X = df[predictors]
    y = df["y"]

    # Train model
    model = RandomForestRegressor(n_estimators=20)
    model.fit(X, y)

    # Disable plot rendering
    plt.show = lambda: None

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Run function
    top_feats = interaction_effects(
        model=model,
        test_df=df,
        predictors=predictors,
        top_n=2
    )

    # Restore stdout
    sys.stdout = sys.__stdout__

    output = captured_output.getvalue()

    # Assertions
    assert isinstance(top_feats, list)
    assert len(top_feats) == 2
    assert all(feat in predictors for feat in top_feats)
    assert "--- Top Interaction Analysis for:" in output
