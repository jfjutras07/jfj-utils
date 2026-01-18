import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from EDA.classification import logistic_regression

#--- Test: logistic_regression ---
def test_logistic_regression():
    """
    Test logistic_regression for standard execution, hyperparameter tuning,
    and the 'for_stacking' pipeline bypass.
    """
    # 1. Setup synthetic binary classification data
    np.random.seed(42)
    data = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.randint(0, 2, 100)
    })
    
    train_df = data.iloc[:80]
    test_df = data.iloc[80:]
    predictors = ["feature1", "feature2"]
    outcome = "target"

    # 2. Test Standard Mode (Optimization + Prediction)
    model = logistic_regression(
        train_df, 
        test_df, 
        outcome=outcome, 
        predictors=predictors, 
        cv=2
    )

    # Check if the returned object is the best estimator (Pipeline)
    assert isinstance(model, Pipeline)
    assert 'model' in model.named_steps
    assert 'scaler' in model.named_steps
    
    # Verify it can predict
    sample_pred = model.predict(test_df[predictors])
    assert len(sample_pred) == len(test_df)

    # 3. Test 'for_stacking' Mode
    # In this mode, it should return the base pipeline without fitting or grid search
    stack_pipe = logistic_regression(
        train_df, 
        test_df, 
        outcome=outcome, 
        predictors=predictors, 
        for_stacking=True
    )

    assert isinstance(stack_pipe, Pipeline)
    # Check that it hasn't been fitted yet (attribute check)
    assert not hasattr(stack_pipe.named_steps['model'], "classes_")

    # 4. Test with Missing Values
    # The internal SimpleImputer should handle this
    df_missing = train_df.copy()
    df_missing.iloc[0, 0] = np.nan
    
    model_missing = logistic_regression(
        df_missing, 
        test_df, 
        outcome=outcome, 
        predictors=predictors, 
        cv=2
    )
    assert model_missing is not None
