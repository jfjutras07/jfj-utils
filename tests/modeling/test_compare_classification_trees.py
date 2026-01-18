import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from modeling.classification_trees import compare_classification_tree_models

#--- Test: compare_classification_tree_models ---
def test_compare_classification_tree_models():
    """
    Test the orchestration of multiple tree models, ensuring the selection 
    of the best model and correct feature importance extraction.
    """
    # 1. Setup synthetic data
    # Creating a simple linear relationship to ensure models can learn something
    np.random.seed(42)
    X = np.random.rand(200, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    
    cols = [f"feat_{i}" for i in range(5)]
    df = pd.DataFrame(X, columns=cols)
    df['target'] = y
    
    train_df = df.iloc[:150]
    test_df = df.iloc[150:]
    
    # 2. Run comparison
    # Note: This assumes underlying functions (xgboost_classification, etc.) 
    # are available in the namespace or mocked.
    winner_model = compare_classification_tree_models(
        train_df, 
        test_df, 
        outcome='target', 
        predictors=cols, 
        cv=2
    )

    # 3. Assertions
    # Ensure the winner is a fitted scikit-learn Pipeline
    assert isinstance(winner_model, Pipeline)
    assert 'model' in winner_model.named_steps
    
    # Check if the model has been fitted (feature_importances_ should exist)
    actual_model = winner_model.named_steps['model']
    assert hasattr(actual_model, "feature_importances_")
    assert len(actual_model.feature_importances_) == len(cols)

    # 4. Data Consistency Check
    # Verify that the test set predictions match the expected shape
    y_pred = winner_model.predict(test_df[cols])
    assert y_pred.shape[0] == test_df.shape[0]
    
    # 5. Type Check
    # Ensure the returned pipeline can handle both NumPy and DataFrame inputs
    assert winner_model.predict(X[0:1]) in [0, 1]
