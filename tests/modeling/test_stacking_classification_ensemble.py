import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from modeling.stacking import stacking_classification_ensemble

#--- Test: stacking_classification_ensemble ---
def test_stacking_classification_ensemble():
    """
    Test the stacking ensemble orchestration, ensuring base models are correctly
    integrated and the meta-model produces valid classification outputs.
    """
    # 1. Setup synthetic data
    np.random.seed(42)
    X = np.random.rand(200, 4)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    
    cols = ["f1", "f2", "f3", "f4"]
    df = pd.DataFrame(X, columns=cols)
    df['target'] = y
    
    train_df = df.iloc[:150]
    test_df = df.iloc[150:]

    # 2. Define base estimators
    # Using simple models for fast unit testing
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]

    # 3. Run Stacking with default final_estimator (LogisticRegression)
    model = stacking_classification_ensemble(
        train_df, 
        test_df, 
        outcome='target', 
        predictors=cols, 
        base_estimators=base_estimators,
        cv=2
    )

    # 4. Assertions
    assert isinstance(model, StackingClassifier)
    assert hasattr(model, "final_estimator_")
    
    # Check if the default meta-model is LogisticRegression
    assert isinstance(model.final_estimator_, LogisticRegression)

    # 5. Prediction Verification
    y_pred = model.predict(test_df[cols])
    assert y_pred.shape[0] == test_df.shape[0]
    assert np.all(np.isin(y_pred, [0, 1]))

    # 6. Test with custom final_estimator
    custom_meta = RandomForestClassifier(n_estimators=5)
    model_custom = stacking_classification_ensemble(
        train_df, 
        test_df, 
        outcome='target', 
        predictors=cols, 
        base_estimators=base_estimators,
        final_estimator=custom_meta,
        cv=2
    )
    assert isinstance(model_custom.final_estimator_, RandomForestClassifier)
