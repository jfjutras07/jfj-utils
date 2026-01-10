import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from eda.explainability import shap_analysis

#--- Function : test_shap_analysis_basic ---
def test_shap_analysis_basic():
    # Create synthetic dataset
    train_df = pd.DataFrame({
        "y": np.random.rand(50),
        "x1": np.random.rand(50),
        "x2": np.random.rand(50)
    })

    test_df = pd.DataFrame({
        "y": np.random.rand(10),
        "x1": np.random.rand(10),
        "x2": np.random.rand(10)
    })

    predictors = ["x1", "x2"]

    X_train = train_df[predictors]
    y_train = train_df["y"]

    # Train model
    model = RandomForestRegressor(n_estimators=20)
    model.fit(X_train, y_train)

    # Run SHAP analysis
    shap_values = shap_analysis(
        model=model,
        train_df=train_df,
        test_df=test_df,
        predictors=predictors
    )

    # Assertions
    assert shap_values is not None
