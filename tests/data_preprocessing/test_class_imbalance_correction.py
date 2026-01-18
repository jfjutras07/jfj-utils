import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing.class_imbalance import class_imbalance_correction

#--- Test: class_imbalance_correction ---
def test_class_imbalance_correction():
    """
    Test the class_imbalance_correction function across different imbalance ratios 
    and dataset sizes to verify the automated strategy selection.
    """
    
    # 1. Test Moderate Imbalance: Should trigger Standard SMOTE (Ratio ~15%)
    # Logic: 0.05 <= minority_ratio < 0.25
    df_mod = pd.DataFrame({
        "feature1": np.random.rand(200),
        "feature2": np.random.rand(200),
        "target": [1] * 30 + [0] * 170
    })
    res_mod = class_imbalance_correction(df_mod, target_col='target', classifier=RandomForestClassifier())
    
    assert res_mod['strategy_used'] == "Standard SMOTE (Moderate Imbalance)"
    assert res_mod['balanced_accuracy'] >= 0
    assert 'pipeline' in res_mod

    # 2. Test Severe Imbalance: Should trigger SMOTE + Tomek (Ratio < 5%)
    # Logic: minority_ratio < 0.05
    # Use enough samples so SMOTE doesn't fail due to lack of neighbors
    df_sev = pd.DataFrame({
        "feature1": np.random.rand(400),
        "feature2": np.random.rand(400),
        "target": [1] * 10 + [0] * 390
    })
    res_sev = class_imbalance_correction(df_sev, target_col='target', classifier=RandomForestClassifier())
    
    assert res_sev['strategy_used'] == "SMOTE + Tomek Links (Severe Imbalance)"

    # 3. Test Balanced Data: Should trigger "None" (Ratio >= 25%)
    # Logic: minority_ratio >= 0.25
    df_bal = pd.DataFrame({
        "feature1": np.random.rand(100),
        "target": [1] * 40 + [0] * 60
    })
    res_bal = class_imbalance_correction(df_bal, target_col='target', classifier=RandomForestClassifier())
    
    assert res_bal['strategy_used'] == "None (Balanced enough)"

    # 4. Test Big Data Logic: Should trigger Random Under-Sampling (n > 500,000)
    # Logic: n_samples > 500,000
    df_big = pd.DataFrame({
        "feature1": np.zeros(500005),
        "target": [1] * 100000 + [0] * 400005
    })
    res_big = class_imbalance_correction(df_big, target_col='target', classifier=RandomForestClassifier())
    
    assert res_big['strategy_used'] == "Random Under-Sampling (Big Data)"
    
    # 5. Pipeline Functional Test
    # Verify the returned pipeline can actually predict on new data
    sample_input = pd.DataFrame({"feature1": [0.5], "feature2": [0.5]})
    prediction = res_mod['pipeline'].predict(sample_input)
    assert len(prediction) == 1
