import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from eda.explainability import pdp_plots

#--- Function : test_pdp_plots_basic ---
def test_pdp_plots_basic():
    # Create synthetic dataset
    df = pd.DataFrame({
        "y": np.random.rand(50),
        "x1": np.random.rand(50),
        "x2": np.random.rand(50)
    })

    predictors = ["x1", "x2"]
    target_features = ["x1"]

    X = df[predictors]
    y = df["y"]

    # Train model
    model = RandomForestRegressor(n_estimators=20)
    model.fit(X, y)

    # Run PDP
    display = pdp_plots(
        model=model,
        train_df=df,
        predictors=predictors,
        target_features=target_features
    )

    # Assertions
    assert display is not None
    assert hasattr(display, "axes_")
