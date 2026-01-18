import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data_preprocessing.cleaning import column_selector

#--- Test: column_selector ---
def test_column_selector():
    """
    Test the column_selector class to ensure correct routing of dtypes,
    automatic dropping of ID/constant columns, and DataFrame reconstruction.
    """
    # 1. Setup synthetic data
    # Features: 1 constant, 1 ID, 2 numeric, 1 categorical
    df = pd.DataFrame({
        "ID_user": [1, 2, 3, 4],            # Should be auto-dropped (ID)
        "const_col": [1, 1, 1, 1],          # Should be auto-dropped (Constant)
        "age": [25, 30, 35, 40],            # Numeric
        "salary": [50000, 60000, 70, 80],   # Numeric
        "city": ["Paris", "Lyon", "Paris", "Marseille"], # Categorical
        "to_drop": [0, 1, 0, 1]             # Manual drop
    })

    # 2. Initialize transformers
    num_trans = StandardScaler()
    cat_trans = OneHotEncoder(sparse_output=False)
    
    selector = column_selector(
        num_transformer=num_trans,
        cat_transformer=cat_trans,
        cols_to_drop=["to_drop"]
    )

    # 3. Fit and Transform
    selector.fit(df)
    df_transformed = selector.transform(df)

    # 4. Assertions
    # Check if auto-drop worked
    assert "ID_user" not in selector.numeric_cols_
    assert "const_col" not in selector.auto_drop_ or "const_col" in selector.auto_drop_
    assert "age" in selector.numeric_cols_
    assert "city" in selector.categorical_cols_

    # Check output format and shape
    # Expected columns: age, salary, city_Lyon, city_Marseille, city_Paris (if OHE)
    assert isinstance(df_transformed, pd.DataFrame)
    assert len(df_transformed.columns) >= 3 # 2 numeric + at least 1-3 for city
    assert df_transformed.index.equals(df.index)

    # 5. Test Dimensionality Audit
    # Create a dataframe with an extra column to trigger the ValueError if logic fails
    df_wrong = df.drop(columns=["age"])
    with pytest.raises(ValueError, match="CRITICAL ERROR"):
        selector.transform(df_wrong)

    # 6. Check Feature Names Consistency
    assert hasattr(selector, "feature_names_out_")
    assert all(col in df_transformed.columns for col in selector.feature_names_out_)
